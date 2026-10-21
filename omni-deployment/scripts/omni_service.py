#!/usr/bin/env python3
"""Launch the pinned FastVideo server with a private, bounded Ray runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

FASTVIDEO_COMMIT = "cdda1b7535fe32699335e39db82c1d4db21ceb88"


def private_ray_node_address(ray) -> str:
    """Match FastVideo's placement constraint to Ray's actual local node."""
    node_id = ray.get_runtime_context().get_node_id()
    nodes = [node for node in ray.nodes() if node.get("Alive")
             and node.get("NodeID") == node_id]
    if len(nodes) != 1:
        raise RuntimeError("Private Ray current node is missing or ambiguous")
    node = nodes[0]
    address = node.get("NodeManagerAddress")
    if not isinstance(address, str) or not address or node.get("Resources", {}).get(f"node:{address}", 0) <= 0:
        raise RuntimeError("Private Ray node address has no matching scheduling resource")
    return address


def main() -> None:
    # Uvicorn replaces this while serving. During imports/model/Ray startup,
    # convert TERM into Python unwinding so the private Ray finally still runs.
    def stop_requested(_signum, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != FASTVIDEO_COMMIT:
        raise RuntimeError(f"FastVideo commit mismatch: {actual}")
    from omni_reference_patch import inspect_patch
    raw_config = json.loads(Path(args.config).read_text())
    policy = raw_config["generator"]["pipeline"]["experimental"].get("h3_reference_image_resize_policy", "native")
    inspect_patch(source, required=policy != "native")
    # The checkout is explicit; a different editable install must not win.
    sys.path.insert(0, str(source))
    from fastvideo.entrypoints.cli.inference_config import build_serve_config
    from fastvideo.api.compat import generator_config_to_fastvideo_args
    typed = build_serve_config(argparse.Namespace(config=args.config))
    generator_config_to_fastvideo_args(typed.generator)
    if args.validate_only:
        print("Pinned ServeConfig and GeneratorConfig conversion: PASS", flush=True)
        return

    import fastvideo
    if not Path(fastvideo.__file__).resolve().is_relative_to(source):
        raise RuntimeError("FastVideo imported from a different source checkout")
    private_ray = None
    try:
        if typed.generator.engine.execution_backend == "ray":
            import ray
            if ray.is_initialized():
                raise RuntimeError("Refusing to reuse an unexpected Ray runtime")
            # Force a new local instance; never attach to RAY_ADDRESS or ray://.
            os.environ.pop("RAY_ADDRESS", None)
            # Ray normalizes loopback to a local interface. Choose that address
            # before ray.init so the driver AND its child workers inherit it.
            os.environ.pop("FASTVIDEO_HOST_IP", None)
            from fastvideo.utils import get_ip
            node_address = get_ip()
            os.environ["FASTVIDEO_HOST_IP"] = node_address
            private_ray = ray
            ray.init(
                address="local",
                num_cpus=int(os.environ.get("OMNI_RAY_CPUS", "8")),
                num_gpus=typed.generator.engine.num_gpus,
                object_store_memory=int(os.environ.get("OMNI_RAY_OBJECT_STORE_BYTES", "4294967296")),
                include_dashboard=False,
                namespace=f"h3-omni-{args.owner_token}",
                _temp_dir=f"/tmp/h3-omni-{os.getuid()}-{args.owner_token[:8]}",
                _node_ip_address=node_address,
            )
            # Ray may normalize loopback to the local interface address.
            # FastVideo pins its first placement bundle to node:<get_ip()>.
            if private_ray_node_address(ray) != node_address:
                raise RuntimeError("Private Ray normalized the chosen node address unexpectedly")
            from omni_ray_health import install_ray_health_adapter
            install_ray_health_adapter()
            resources = ray.cluster_resources()
            if int(resources.get("GPU", 0)) != typed.generator.engine.num_gpus:
                raise RuntimeError(f"Unexpected private Ray GPU resources: {resources}")
            print(json.dumps({"private_ray": True, "node_resource_verified": True,
                              "gpu_resources": resources.get("GPU"),
                              "cpu_resources": resources.get("CPU")}), flush=True)

        from fastvideo.entrypoints.cli.serve import ServeSubcommand
        ServeSubcommand().cmd(argparse.Namespace(config=args.config))
    finally:
        if private_ray is not None:
            # This only shuts down the Ray instance created by this process.
            private_ray.shutdown()


if __name__ == "__main__":
    main()
