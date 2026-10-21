#!/usr/bin/env python3
"""Identity-preserving wrapper for pinned native ComfyUI (no custom nodes)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import sys

from comfy_control import command_arguments, validate_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--owner-token", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    if config.get("owner_token") != args.owner_token:
        raise RuntimeError("launch_owner_token_mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(config["gpu"]):
        raise RuntimeError("single_gpu_visibility_mismatch")

    def stopping(_signum, _frame):
        # ComfyUI handles KeyboardInterrupt around the event loop. During
        # imports/startup the outer handler still exits this owned wrapper.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stopping)
    source = Path(config["source"])
    os.chdir(source)
    sys.path.insert(0, str(source))
    # Updating Python's argv leaves /proc/cmdline's wrapper/token identity intact.
    sys.argv = command_arguments(config)
    try:
        runpy.run_path(str(source / "main.py"), run_name="__main__")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
