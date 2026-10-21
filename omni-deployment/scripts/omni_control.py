#!/usr/bin/env python3
"""Own one FastH3 Omni service: explicit configuration, PID identity, private Ray.

This script never kills arbitrary GPU processes, invokes ray stop, changes the
driver, or mutates an existing FastVideo installation. Lifecycle requires Linux.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
PIN = "cdda1b7535fe32699335e39db82c1d4db21ceb88"
RECIPE_ENV = {
    "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
    "FASTVIDEO_H3_VSA_FP4": "1",
    "FASTVIDEO_H3_VSA_FP4_PREFIX": "fp8",
    "FASTVIDEO_H3_VSA_FP4_TWO_LEVEL_P": "1",
    "FASTVIDEO_H3_VSA_FP4_K_SMOOTH": "1",
    "FASTVIDEO_H3_GATE_CHUNK_ROWS": "8192",
    "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
    "FASTVIDEO_VSA_TRITON": "1",
    "FASTVIDEO_VSA_PTX_BLACKWELL": "0",
    "FASTVIDEO_FA4": "0",
    "FASTVIDEO_ULYSSES_A2A": "auto",
    "FASTVIDEO_MINIMAX_H3_EXACT_KERNELS": "all",
    "FASTVIDEO_H3_VSA_HEADS_FIRST_TILE": "1",
    "FASTVIDEO_H3_VAE_TILE_PARALLEL": "1",
    "FASTVIDEO_H3_VAE_TILE_BATCH": "4",
    "FASTVIDEO_STAGE_LOGGING": "1",
    "FASTVIDEO_LOG_ALL_PROCESSES": "1",
    "RAY_USAGE_STATS_ENABLED": "0",
    "FASTVIDEO_HOST_IP": "127.0.0.1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
}


def absolute(value: str, name: str, *, follow_symlinks: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return path.resolve() if follow_symlinks else path


def toolchain_environment(root: Path, python: Path) -> dict[str, str]:
    # Preserve the venv's executable path: uv's python symlink points outside
    # the venv, whereas the CUDA wheel is installed inside its site-packages.
    venv_bin = python.parent
    cuda_home = absolute(os.environ.get("OMNI_CUDA_HOME", str(
        venv_bin.parent / "lib/python3.12/site-packages/nvidia/cu13")), "OMNI_CUDA_HOME")
    media_bin = absolute(os.environ.get("OMNI_MEDIA_BIN", str(root / ".tools/media/bin")), "OMNI_MEDIA_BIN")
    prefix = [str(venv_bin), str(cuda_home / "bin"), str(media_bin)]
    inherited_path = os.environ.get("PATH") or os.defpath
    return {"CUDA_HOME": str(cuda_home), "CUDACXX": str(cuda_home / "bin/nvcc"),
            "PATH": os.pathsep.join([*prefix, inherited_path])}


def settings(profile: str) -> tuple[dict, dict, dict]:
    root = absolute(os.environ.get("H3_OMNI_ROOT", "/home/isaac/workdir/h3"), "H3_OMNI_ROOT")
    values = {}
    for name in ("OMNI_MODEL_DIR", "OMNI_ADALN_TABLE", "OMNI_DECODER"):
        if not os.environ.get(name):
            raise ValueError(f"Set {name} to the verified asset's absolute path")
        values[name] = str(absolute(os.environ[name], name))
    for name, default in (("OMNI_INPUT_DIR", root / "inputs"),
                          ("OMNI_OUTPUT_DIR", root / "outputs/omni")):
        values[name] = str(absolute(os.environ.get(name, str(default)), name))
    template = HERE.parent / "configs" / f"fasth3-omni-{profile}.template.json"
    config = json.loads(template.read_text())

    def replace(obj):
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            return values[obj[2:-1]]
        if isinstance(obj, list):
            return [replace(v) for v in obj]
        if isinstance(obj, dict):
            return {k: replace(v) for k, v in obj.items()}
        return obj

    config = replace(config)
    flags = {}
    for name in ("OMNI_VAE_OFFLOAD", "OMNI_EXPANDABLE_SEGMENTS"):
        value = os.environ.get(name, "0")
        if value not in {"0", "1"}:
            raise ValueError(f"{name} must be 0 or 1")
        flags[name] = value == "1"
    config["generator"]["engine"]["offload"]["vae"] = flags["OMNI_VAE_OFFLOAD"]
    port = int(os.environ.get("OMNI_PORT", "8108"))
    if not 1024 <= port <= 65535:
        raise ValueError("OMNI_PORT must be between 1024 and 65535")
    config["server"]["port"] = port
    env = dict(RECIPE_ENV)
    env["PYTORCH_ALLOC_CONF"] = ("expandable_segments:True" if flags["OMNI_EXPANDABLE_SEGMENTS"]
                                 else "backend:native")
    env.update({"FASTVIDEO_H3_ADALN_TABLE": values["OMNI_ADALN_TABLE"],
                "FASTVIDEO_H3_SP_PROFILE": os.environ.get("OMNI_SP_PROFILE", "0"),
                "FASTVIDEO_H3_VAE_NVFP4_DECODER": values["OMNI_DECODER"],
                "FASTVIDEO_H3_REF2VA_MEMO_ENTRIES": os.environ.get("OMNI_MEMO_ENTRIES", "16"),
                "OMNI_RAY_CPUS": os.environ.get("OMNI_RAY_CPUS", "8"),
                "OMNI_RAY_OBJECT_STORE_BYTES": os.environ.get("OMNI_RAY_OBJECT_STORE_BYTES", "4294967296"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("OMNI_GPU_IDS", "1" if profile == "sp1" else "0,1")})
    if env["CUDA_VISIBLE_DEVICES"] not in ({"0", "1"} if profile == "sp1" else {"0,1", "1,0"}):
        raise ValueError("SP1 requires GPU 0 or 1; SP2 requires both 0 and 1")
    if env["FASTVIDEO_H3_SP_PROFILE"] not in {"0", "1"}:
        raise ValueError("OMNI_SP_PROFILE must be 0 or 1")
    for key in ("FASTVIDEO_H3_REF2VA_MEMO_ENTRIES", "OMNI_RAY_CPUS", "OMNI_RAY_OBJECT_STORE_BYTES"):
        value = int(env[key])
        if value < (0 if key.endswith("ENTRIES") else 1):
            raise ValueError(f"Invalid {key}")
    paths = {"root": str(root), "run": str(root / "run/omni"),
             "python": str(absolute(os.environ.get("OMNI_PYTHON", str(root / "envs/omni/bin/python")), "OMNI_PYTHON", follow_symlinks=False)),
             "source": str(absolute(os.environ.get("OMNI_FASTVIDEO_SRC", str(root / "src/FastVideo")), "OMNI_FASTVIDEO_SRC"))}
    env.update(toolchain_environment(root, Path(paths["python"])))
    return config, env, paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_identity(config: dict, env: dict) -> dict:
    model = Path(config["generator"]["model_path"])
    contract_path = model / "fastvideo_inference.json"
    contract = json.loads(contract_path.read_text())
    if contract.get("model_type") != "ref2va" or contract.get("num_inference_steps") != 8:
        raise ValueError("Model contract must be Omni Ref2VA with 8 forwards")
    for name in ("transformer_ref", "scheduler", "audio_scheduler", "text_encoder", "processor", "tokenizer", "vae", "audio_vae"):
        if not (model / name).is_dir():
            raise FileNotFoundError(f"Missing complete model component: {model / name}")
    if not (model / "transformer_ref/nvfp4_weights.safetensors").is_file():
        raise FileNotFoundError("Packed NVFP4 transformer weights are absent")
    identity = {"contract_sha256": sha256(contract_path), "files": {}}
    for name in ("FASTVIDEO_H3_ADALN_TABLE", "FASTVIDEO_H3_VAE_NVFP4_DECODER"):
        path = Path(env[name])
        if not path.is_file():
            raise FileNotFoundError(path)
        identity["files"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if os.environ.get("OMNI_ASSET_MANIFEST"):
        path = absolute(os.environ["OMNI_ASSET_MANIFEST"], "OMNI_ASSET_MANIFEST")
        identity["asset_manifest"] = {"path": str(path), "sha256": sha256(path)}
    return identity


def atomic_json(path: Path, data: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.chmod(temp, 0o600)
    temp.replace(path)


@contextmanager
def locked(run: Path):
    run.mkdir(parents=True, exist_ok=True)
    with (run / "control.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def proc_identity(pid: int) -> dict | None:
    try:
        proc = Path(f"/proc/{pid}")
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"start_ticks": fields[19], "pgid": int(fields[2]), "uid": proc.stat().st_uid,
                "cmdline": (proc / "cmdline").read_bytes().split(b"\0")}
    except FileNotFoundError:
        return None


def owned(state: dict) -> bool:
    info = proc_identity(state["pid"])
    if info is None:
        return False
    expected = (info["start_ticks"] == state["start_ticks"] and info["pgid"] == state["pid"]
                and info["uid"] == os.getuid() == state["uid"]
                and state["service_script"].encode() in info["cmdline"]
                and state["owner_token"].encode() in info["cmdline"])
    if not expected:
        raise RuntimeError("PID ownership mismatch; refusing reuse or signals")
    return True


def probe(state: dict) -> dict:
    base = f"http://127.0.0.1:{state['port']}"
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=3) as response:
            models = json.load(response).get("data", [])
        if not any(m.get("id") == state["model_name"] and m.get("root") == state["model_path"] for m in models):
            return {"status": "IDENTITY_MISMATCH"}
        with urllib.request.urlopen(base + "/health", timeout=3) as response:
            health = json.load(response)
        return {"status": "READY" if health.get("status") == "ok" else "UNHEALTHY"}
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {"status": "STARTING_OR_UNHEALTHY", "detail": str(exc)}


def read_state(run: Path) -> dict | None:
    path = run / "service.json"
    return json.loads(path.read_text()) if path.exists() else None


def child_environment(env: dict) -> dict:
    child_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("FASTVIDEO_", "RAY_", "NCCL_"))
                 and k not in {"PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"}}
    child_env.update(env)
    return child_env


def validate(config_path: Path, paths: dict, env: dict) -> None:
    subprocess.run([paths["python"], str(HERE / "omni_service.py"), "--config", str(config_path),
                    "--source", paths["source"], "--owner-token", "validation", "--validate-only"],
                   check=True, env=child_environment(env), cwd=paths["source"])


def start(profile: str, run: Path) -> dict:
    config, env, paths = settings(profile)
    identity = {"config": config, "env": env, "paths": paths, "pin": PIN,
                "assets": asset_identity(config, env), "launcher_sha256": sha256(HERE / "omni_service.py"),
                "health_adapter_sha256": sha256(HERE / "omni_ray_health.py")}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    name = f"fasth3-omni-qad-{profile}-{digest[:12]}"
    config["server"]["served_model_name"] = name
    previous = read_state(run)
    if previous and owned(previous):
        if previous["fingerprint"] != digest:
            raise RuntimeError("An owned service with a different configuration is active; stop it explicitly first")
        return {"reused": True, **previous, **probe(previous)}
    with socket.socket() as test:
        test.bind(("127.0.0.1", config["server"]["port"]))
    for folder in (Path(config["server"]["output_dir"]), Path(config["server"]["media_roots"][0]), Path(paths["root"]) / "logs/omni"):
        folder.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    config_path = run / f"{profile}-{digest[:12]}.json"
    atomic_json(config_path, config)
    validate(config_path, paths, env)
    log_path = Path(paths["root"]) / "logs/omni" / f"{time.strftime('%Y%m%d-%H%M%S')}-{profile}-{token[:8]}.log"
    with log_path.open("x") as log, open(os.devnull, "rb") as null:
        log.write(json.dumps({"identity": identity, "owner_token": token, "config": str(config_path)}) + "\n")
        log.flush()
        process = subprocess.Popen([paths["python"], "-u", str(HERE / "omni_service.py"), "--config", str(config_path),
                                    "--source", paths["source"], "--owner-token", token],
                                   stdin=null, stdout=log, stderr=subprocess.STDOUT, cwd=paths["source"],
                                   env=child_environment(env), start_new_session=True, close_fds=True)
    info = None
    for _ in range(50):
        if process.poll() is not None:
            raise RuntimeError(f"Service exited during startup; inspect {log_path}")
        info = proc_identity(process.pid)
        if info and token.encode() in info["cmdline"]:
            break
        time.sleep(0.02)
    if not info or token.encode() not in info["cmdline"]:
        process.terminate()
        raise RuntimeError("Could not establish child process identity")
    state = {"pid": process.pid, "pgid": process.pid, "uid": os.getuid(), "start_ticks": info["start_ticks"],
             "owner_token": token, "service_script": str(HERE / "omni_service.py"), "profile": profile,
             "fingerprint": digest, "model_name": name, "model_path": config["generator"]["model_path"],
             "port": config["server"]["port"], "config": str(config_path), "log": str(log_path),
             "started_at": time.time()}
    atomic_json(run / "service.json", state)
    return {"status": "STARTING", **state}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "validate", "start", "status", "stop"))
    parser.add_argument("profile", nargs="?", choices=("sp1", "sp2"), default="sp1")
    parser.add_argument("--force", action="store_true", help="After ownership checks, kill only this service's process group")
    parser.add_argument("--stop-timeout", type=int, default=90)
    args = parser.parse_args()
    root = absolute(os.environ.get("H3_OMNI_ROOT", "/home/isaac/workdir/h3"), "H3_OMNI_ROOT")
    run = root / "run/omni"
    if args.action == "render":
        config, env, paths = settings(args.profile)
        print(json.dumps({"config": config, "environment": env, "paths": paths}, indent=2))
        return
    if args.action in ("start", "stop", "status") and sys.platform != "linux":
        raise RuntimeError("Service lifecycle commands require Linux /proc identity checks")
    with locked(run):
        if args.action == "validate":
            config, env, paths = settings(args.profile)
            config_path = run / f"validate-{args.profile}.json"
            atomic_json(config_path, config)
            validate(config_path, paths, env)
            return
        if args.action == "start":
            result = start(args.profile, run)
        else:
            state = read_state(run)
            if not state or not owned(state):
                result = {"status": "STOPPED", "state": state}
            elif args.action == "status":
                result = {**state, **probe(state)}
            else:
                if args.force:
                    os.killpg(state["pgid"], signal.SIGKILL)
                else:
                    os.kill(state["pid"], signal.SIGTERM)
                deadline = time.monotonic() + max(args.stop_timeout, 1)
                while owned(state) and time.monotonic() < deadline:
                    time.sleep(0.25)
                if owned(state):
                    raise RuntimeError("Owned service still stopping; inspect its log before retrying stop --force")
                # Retain the complete last lifecycle record instead of deleting it.
                state["stopped_at"] = time.time()
                atomic_json(run / f"stopped-{state['owner_token']}.json", state)
                (run / "service.json").unlink()
                result = {"status": "STOPPED", "last_state": state}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
