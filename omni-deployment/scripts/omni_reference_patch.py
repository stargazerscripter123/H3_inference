#!/usr/bin/env python3
"""Check or explicitly apply the pinned reference-image geometry patch."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

PATCH_DIR = Path(__file__).resolve().parents[1] / "patches"
MANIFEST = PATCH_DIR / "reference-image-cap-v1.json"
PATCH = PATCH_DIR / "reference-image-cap-v1.patch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_patch(source: Path, *, required: bool = False) -> dict:
    source = source.resolve()
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema_version") != 1 or sha256(PATCH) != manifest["patch_sha256"]:
        raise RuntimeError("Reference geometry patch manifest/hash mismatch")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != manifest["upstream_commit"]:
        raise RuntimeError("Reference geometry patch requires the pinned FastVideo commit")
    files = {}
    states = set()
    for relative, expected in manifest["files"].items():
        path = source / relative
        if not path.resolve().is_relative_to(source) or path.is_symlink():
            raise RuntimeError("Unsafe path in reference geometry patch")
        digest = sha256(path)
        state = ("native" if digest == expected["before_sha256"] else
                 "patched" if digest == expected["after_sha256"] else "modified")
        states.add(state)
        files[relative] = digest
    if len(states) != 1 or states == {"modified"}:
        raise RuntimeError("Reference geometry source is modified or partially patched; refusing to overwrite it")
    state = states.pop()
    if required and state != "patched":
        raise RuntimeError("Reference image cap requires explicit omni_reference_patch.py apply first")
    return {"state": state, "upstream_commit": head, "manifest_sha256": sha256(MANIFEST),
            "patch_sha256": manifest["patch_sha256"], "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "apply"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    source, root = args.source.resolve(), args.project_root.resolve()
    if not source.is_relative_to(root):
        raise ValueError("Source checkout must belong to the explicit project root")
    run = root / "run/omni"
    run.mkdir(parents=True, exist_ok=True)
    with (run / "control.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = inspect_patch(source)
        if args.action == "apply" and state["state"] != "patched":
            if (run / "service.json").exists():
                raise RuntimeError("Stop the owned service cleanly before patching its source checkout")
            subprocess.run(["git", "-C", str(source), "apply", "--check", str(PATCH)], check=True)
            subprocess.run(["git", "-C", str(source), "apply", str(PATCH)], check=True)
            state = inspect_patch(source, required=True)
        print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
