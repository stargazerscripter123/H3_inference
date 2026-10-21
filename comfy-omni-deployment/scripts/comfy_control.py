#!/usr/bin/env python3
"""Own one isolated, single-GPU ComfyUI Omni service on Linux.

No global process cleanup is performed. Stop always requires the owner token,
UID, process group, start ticks and wrapper argv to match the saved service.
READY means the HTTP server and native reference node are ready; models are
loaded lazily by the first submitted graph. Model SHA checks belong to the
comparison client; this controller checks pinned model names and file sizes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
PIN = "b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c"
VERSION = "0.42.0"
PORT = 8188
READINESS = "Owned HTTP listener and native node; weights load on first request"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("project_root_must_be_absolute")
    root = path.resolve()
    if root == Path("/"):
        raise ValueError("project_root_cannot_be_filesystem_root")
    return root


def confined(value: str | Path, roots: tuple[Path, ...], *, allow_root: bool = True) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("directory_must_be_absolute")
    path = path.resolve()
    for root in roots:
        root = root.resolve()
        if path.is_relative_to(root) and (allow_root or path != root):
            return path
    raise ValueError("directory_outside_allowed_project_subtree")


def configuration(root: Path, gpu: int = 1, input_dir: str | None = None,
                  output_dir: str | None = None) -> dict:
    if type(gpu) is not int or gpu not in (0, 1):
        raise ValueError("gpu_must_be_single_physical_device_0_or_1")
    root = project_root(root)
    run = confined(root / "run/comfy-omni", (root,), allow_root=False)
    input_roots = tuple(confined(path, (root,), allow_root=False)
                        for path in (root / "inputs", run / "input"))
    output_base = confined(root / "outputs/comfy-omni", (root,), allow_root=False)
    input_dir = str(confined(input_dir or run / "input", input_roots))
    output_dir = str(confined(output_dir or output_base, (output_base,)))
    source = confined(root / "src/ComfyUI-Omni", (root,), allow_root=False)
    models = confined(source / "models", (source,), allow_root=False)
    # Keep the venv spelling; its Python executable can be a managed-Python symlink.
    venv = root / "envs/comfy-omni"
    if venv.resolve() != venv:
        raise ValueError("isolated_venv_directory_must_not_be_aliased")
    python = venv / "bin/python"
    config = {"schema_version": 1, "project_root": str(root), "source": str(source),
              "python": str(python), "comfy_commit": PIN, "comfy_version": VERSION,
              "listen": "127.0.0.1", "port": PORT, "gpu": gpu,
              "input_dir": input_dir, "output_dir": output_dir,
              "models_dir": str(models), "run_dir": str(run),
              "user_dir": str(confined(run / "user", (run,), allow_root=False)),
              "temp_base": str(confined(run / "temp", (run,), allow_root=False)),
              "disable_metadata": True, "disable_all_custom_nodes": True,
              "disable_auto_launch": True, "single_gpu": True,
              "readiness_scope": READINESS,
              "pins_sha256": sha256(HERE.parent / "pins.json"),
              "wrapper_sha256": sha256(HERE / "comfy_service.py")}
    return config


def validate_config(config: dict) -> None:
    root = project_root(config["project_root"])
    expected = configuration(root, config["gpu"], config["input_dir"], config["output_dir"])
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("launch_configuration_mismatch")
    token = config.get("owner_token")
    if token is not None:
        if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
            raise ValueError("invalid_owner_token")
        if config.get("temp_dir") != str(Path(config["temp_base"]) / token):
            raise ValueError("temp_directory_mismatch")


def command_arguments(config: dict) -> list[str]:
    """The fixed core supports every flag here; no custom-node whitelist exists."""
    return [str(Path(config["source"]) / "main.py"),
            "--listen", "127.0.0.1", "--port", str(PORT),
            "--disable-metadata", "--disable-auto-launch", "--disable-all-custom-nodes",
            "--preview-method", "none", "--input-directory", config["input_dir"],
            "--output-directory", config["output_dir"],
            "--temp-directory", config["temp_dir"], "--user-directory", config["user_dir"],
            "--models-directory", config["models_dir"]]


def child_environment(config: dict) -> dict[str, str]:
    # Allow only OS/session essentials. In particular, no inherited FastVideo,
    # Ray, NCCL, Python/CUDA search paths or allocator/backend switches survive.
    keep = {"HOME", "USER", "LOGNAME", "LANG", "TZ", "TERM"}
    env = {k: v for k, v in os.environ.items() if k in keep or k.startswith("LC_")}
    root = Path(config["project_root"])
    env.update({"PATH": os.pathsep.join([str(Path(config["python"]).parent),
                                        str(root / ".tools/media/bin"), "/usr/local/bin", "/usr/bin", "/bin"]),
                "VIRTUAL_ENV": str(root / "envs/comfy-omni"), "PYTHONNOUSERSITE": "1",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
                "PYTHONUNBUFFERED": "1"})
    return env


def verify_installation(config: dict) -> dict:
    validate_config(config)
    source = Path(config["source"])
    if not Path(config["python"]).is_file() or not (source / "main.py").is_file():
        raise FileNotFoundError("isolated_python_or_comfy_source_missing")
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        raise RuntimeError("comfy_commit_mismatch")
    if subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"],
                      check=False).returncode != 0:
        raise RuntimeError("comfy_tracked_source_modified")
    pins = json.loads((HERE.parent / "pins.json").read_text())
    if pins.get("comfyui", {}).get("commit") != PIN:
        raise ValueError("package_pins_mismatch")
    assets = []
    for model in pins["models"]:
        path = confined(Path(config["models_dir"]) / model["comfy_models_relative_path"],
                        (Path(config["models_dir"]),), allow_root=False)
        if not path.is_file() or path.stat().st_size != model["size_bytes"]:
            raise ValueError("model_file_missing_or_size_mismatch:" + model["component"])
        assets.append({"component": model["component"], "size_bytes": model["size_bytes"],
                       "expected_sha256": model["sha256"]})
    return {"validation": "pinned_source_and_model_sizes; client_verifies_sha256", "models": assets}


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def locked(run: Path):
    run.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(run / "control.lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def proc_identity(pid: int) -> dict | None:
    try:
        proc = Path(f"/proc/{pid}")
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"start_ticks": fields[19], "pgid": int(fields[2]),
                "uid": proc.stat().st_uid,
                "cmdline": (proc / "cmdline").read_bytes().split(b"\0")}
    except (FileNotFoundError, ProcessLookupError):
        return None


def validate_saved_owner(state: dict, *, require_start_ticks: bool = True) -> None:
    if (type(state.get("pid")) is not int or state["pid"] <= 1 or
            state.get("pgid") != state["pid"] or state.get("uid") != os.getuid() or
            state.get("service_script") != str(HERE / "comfy_service.py") or
            not re.fullmatch(r"[0-9a-f]{32}", str(state.get("owner_token", "")))):
        raise RuntimeError("invalid_saved_owner_identity")
    if require_start_ticks and (not isinstance(state.get("start_ticks"), str)
                                or not state["start_ticks"].isdecimal()):
        raise RuntimeError("invalid_saved_owner_identity")


def unverified_alive(state: dict) -> bool:
    """Quarantine a failed bootstrap without pretending it is signal-authorized."""
    validate_saved_owner(state, require_start_ticks=False)
    return proc_identity(state["pid"]) is not None or bool(group_members(state["pgid"]))


def owned(state: dict) -> bool:
    validate_saved_owner(state)
    current = proc_identity(state["pid"])
    if current is None:
        return False
    argv = current["cmdline"]
    expected = [state["service_script"].encode(), b"--config", state["config"].encode(),
                b"--owner-token", state["owner_token"].encode()]
    if (current["uid"] != state["uid"] or current["start_ticks"] != state["start_ticks"] or
            current["pgid"] != state["pgid"] or
            not any(argv[i:i + len(expected)] == expected for i in range(len(argv)))):
        raise RuntimeError("pid_ownership_mismatch_refusing_signals")
    return True


def group_members(pgid: int) -> list[dict]:
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                continue
            info = {"pgid": int(fields[2]), "uid": path.stat().st_uid, "start_ticks": fields[19]}
        except (FileNotFoundError, ProcessLookupError):
            continue
        if info and info["pgid"] == pgid:
            result.append({"pid": int(path.name), "uid": info["uid"], "start_ticks": info["start_ticks"]})
    return result


def listener_owned(state: dict) -> bool:
    """A responding unrelated server must never be reported as this service."""
    address = f"0100007F:{PORT:04X}"
    inodes = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1] == address and fields[3] == "0A":
            inodes.add(f"socket:[{fields[9]}]")
    if not inodes:
        return False
    try:
        for fd in Path(f"/proc/{state['pid']}/fd").iterdir():
            try:
                if os.readlink(fd) in inodes:
                    return True
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        return False
    return False


def probe(state: dict) -> dict:
    if not listener_owned(state):
        return {"status": "STARTING_OR_UNHEALTHY", "readiness_scope": READINESS}
    base = f"http://127.0.0.1:{PORT}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(base + "/system_stats", timeout=3) as response:
            stats = json.load(response)
        if stats.get("system", {}).get("comfyui_version") != VERSION:
            return {"status": "IDENTITY_MISMATCH"}
        devices = stats.get("devices", [])
        if len(devices) != 1 or devices[0].get("type") != "cuda" or devices[0].get("index") != 0:
            return {"status": "IDENTITY_MISMATCH"}
        with opener.open(base + "/object_info/MiniMaxH3ReferenceToVideo", timeout=3) as response:
            nodes = json.load(response)
        if "MiniMaxH3ReferenceToVideo" not in nodes:
            return {"status": "NATIVE_NODE_MISSING"}
        return {"status": "READY", "readiness_scope": READINESS}
    except (OSError, ValueError, urllib.error.URLError):
        return {"status": "STARTING_OR_UNHEALTHY", "readiness_scope": READINESS}


def read_state(run: Path) -> dict | None:
    path = run / "service.json"
    return json.loads(path.read_text()) if path.exists() else None


def check_port() -> None:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", PORT))


def fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def retired(run: Path, state: dict) -> None:
    state = {**state, "stopped_at": time.time()}
    token = state["owner_token"]
    path = run / f"stopped-{token}-{time.time_ns()}.json"
    atomic_json(path, state)
    (run / "service.json").unlink(missing_ok=True)


def start(config: dict, run: Path) -> dict:
    previous = read_state(run)
    digest = fingerprint(config)
    if previous:
        if previous.get("identity_established") is False:
            if unverified_alive(previous):
                raise RuntimeError("unverified_start_requires_review_no_signals_sent")
            retired(run, previous)
        elif owned(previous):
            if previous.get("stop_requested_at"):
                raise RuntimeError("owned_service_is_stopping")
            if previous.get("fingerprint") != digest:
                raise RuntimeError("owned_service_has_different_configuration_stop_explicitly")
            return {**previous, **probe(previous), "reused": True}
        elif group_members(previous["pgid"]):
            raise RuntimeError("orphaned_process_group_requires_review_no_signals_sent")
        else:
            retired(run, previous)
    check_port()
    assets = verify_installation(config)
    token = uuid.uuid4().hex
    launch = {**config, "owner_token": token,
              "temp_dir": str(Path(config["temp_base"]) / token)}
    root = Path(config["project_root"])
    for key in ("input_dir", "output_dir", "temp_dir", "user_dir"):
        Path(launch[key]).mkdir(parents=True, exist_ok=True, mode=0o700)
    log_dir = confined(root / "logs/comfy-omni", (root,), allow_root=False)
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    config_path = run / f"launch-{token}.json"
    atomic_json(config_path, launch)
    log_path = log_dir / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{token}.log"
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    script = str(HERE / "comfy_service.py")
    argv = [config["python"], "-u", script, "--config", str(config_path), "--owner-token", token]
    with os.fdopen(log_fd, "w") as log, open(os.devnull, "rb") as null:
        # Tokens are in the private launch/state files and OS argv, not the log.
        log.write(json.dumps({"comfy_commit": PIN, "fingerprint": digest, "gpu": config["gpu"]}) + "\n")
        log.flush()
        process = subprocess.Popen(argv, cwd=config["source"], env=child_environment(config),
                                   stdin=null, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    state = {"schema_version": 1, "pid": process.pid, "pgid": process.pid, "uid": os.getuid(),
             "owner_token": token, "service_script": script, "config": str(config_path),
             "fingerprint": digest, "comfy_commit": PIN, "comfy_version": VERSION,
             "gpu": config["gpu"], "port": PORT, "input_dir": config["input_dir"],
             "output_dir": config["output_dir"], "models_dir": config["models_dir"],
             "source": config["source"], "python": config["python"],
             "disable_metadata": True, "disable_all_custom_nodes": True,
             "started_at": time.time(), "log": str(log_path), "asset_preflight": assets,
             "readiness_scope": READINESS}
    info = None
    for _ in range(100):
        if process.poll() is not None:
            failure = {**state, "identity_established": False, "status": "EXITED_DURING_START"}
            atomic_json(run / f"failed-start-{token}.json", failure)
            if group_members(process.pid):
                # The wrapper may have spawned a same-group helper before
                # exiting. Retain a blocking record; never claim it is stopped.
                atomic_json(run / "service.json", {**failure, "status": "ORPHANED"})
            raise RuntimeError("service_exited_during_start_see_private_log")
        info = proc_identity(process.pid)
        if info and script.encode() in info["cmdline"] and token.encode() in info["cmdline"]:
            state["start_ticks"] = info["start_ticks"]
            state["identity_established"] = True
            # Establish ownership before any signal can ever be permitted.
            owned(state)
            atomic_json(run / "service.json", state)
            return {**state, "status": "STARTING", "reused": False}
        time.sleep(0.02)
    unknown = {**state, "identity_established": False, "status": "IDENTITY_UNVERIFIED"}
    atomic_json(run / f"unverified-start-{token}.json", unknown)
    atomic_json(run / "service.json", unknown)
    # No terminate() shortcut: absence of verified /proc identity is not authority.
    raise RuntimeError("child_identity_unverified_no_signals_sent_see_private_record")


def status(run: Path) -> dict:
    state = read_state(run)
    if state is None:
        return {"status": "STOPPED"}
    if state.get("identity_established") is False:
        return {**state, "status": "ORPHANED" if unverified_alive(state) else "STOPPED",
                "reason": "bootstrap_identity_was_not_established_no_signals_sent"}
    if not owned(state):
        if group_members(state["pgid"]):
            return {**state, "status": "ORPHANED", "reason": "leader_gone_group_remains_no_signals_sent"}
        return {**state, "status": "STOPPED"}
    if state.get("stop_requested_at"):
        return {**state, "status": "STOPPING"}
    return {**state, **probe(state)}


def stop(run: Path, expected_owner_token: str, *, timeout: float = 90,
         force: bool = False) -> dict:
    state = read_state(run)
    if not state or state.get("owner_token") != expected_owner_token:
        raise RuntimeError("expected_owner_token_mismatch_refusing_stop")
    if state.get("identity_established") is False:
        if unverified_alive(state):
            return {**state, "status": "ORPHANED",
                    "reason": "bootstrap_identity_was_not_established_no_signals_sent"}
        retired(run, state)
        return {**state, "status": "STOPPED"}
    live = owned(state)
    if not live:
        if group_members(state["pgid"]):
            return {**state, "status": "ORPHANED", "reason": "leader_gone_group_remains_no_signals_sent"}
        retired(run, state)
        return {**state, "status": "STOPPED"}
    if force:
        members = group_members(state["pgid"])
        if any(member["uid"] != state["uid"] for member in members):
            raise RuntimeError("process_group_uid_mismatch_refusing_force")
        # Recheck immediately before signalling; never trust an earlier probe.
        if owned(state):
            try:
                os.killpg(state["pgid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        state["force_requested_at"] = time.time()
    elif not state.get("stop_requested_at"):
        if owned(state):
            try:
                os.kill(state["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    state.setdefault("stop_requested_at", time.time())
    atomic_json(run / "service.json", state)
    deadline = time.monotonic() + max(0, timeout)
    while owned(state) and time.monotonic() < deadline:
        time.sleep(0.1)
    if owned(state):
        return {**state, "status": "STOPPING"}
    if group_members(state["pgid"]):
        return {**state, "status": "ORPHANED", "reason": "leader_gone_group_remains_no_signals_sent"}
    retired(run, state)
    return {**state, "status": "STOPPED"}


def sanitized(result: dict) -> dict:
    keys = {"schema_version", "status", "reason", "pid", "pgid", "uid", "start_ticks",
            "fingerprint", "comfy_commit", "comfy_version", "gpu", "port", "input_dir", "output_dir",
            "models_dir", "source", "python", "disable_metadata", "disable_all_custom_nodes",
            "started_at", "stop_requested_at", "force_requested_at", "readiness_scope", "reused"}
    return {key: value for key, value in result.items() if key in keys}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "validate", "start", "status", "stop"))
    parser.add_argument("--project-root", default="/home/isaac/workdir/h3")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-owner-token")
    parser.add_argument("--stop-timeout", type=float, default=90)
    parser.add_argument("--force", action="store_true", help="SIGKILL only the identity-checked owned process group")
    parser.add_argument("--sanitized", action="store_true", help="Output only public metadata; omit owner token and log/config paths")
    args = parser.parse_args(argv)
    if args.action == "stop" and not args.expected_owner_token:
        parser.error("stop requires --expected-owner-token")
    if args.action != "stop" and (args.expected_owner_token or args.force):
        parser.error("owner token and force are only valid for stop")
    if not 0 <= args.stop_timeout <= 600:
        parser.error("--stop-timeout must be between 0 and 600 seconds")
    root = project_root(args.project_root)
    run = confined(root / "run/comfy-omni", (root,), allow_root=False)
    if args.action == "render":
        result = configuration(root, args.gpu, args.input_dir, args.output_dir)
    else:
        if sys.platform != "linux":
            raise RuntimeError("service_lifecycle_requires_linux_proc")
        with locked(run):
            if args.action in ("start", "validate"):
                config = configuration(root, args.gpu, args.input_dir, args.output_dir)
                result = (start(config, run) if args.action == "start"
                          else {"status": "VALIDATED", "asset_preflight": verify_installation(config)})
            elif args.action == "status":
                result = status(run)
            else:
                result = stop(run, args.expected_owner_token, timeout=args.stop_timeout, force=args.force)
    print(json.dumps(sanitized(result) if args.sanitized else result, indent=2))
    return 0 if result.get("status") not in {"ORPHANED", "IDENTITY_MISMATCH", "NATIVE_NODE_MISSING"} else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        # No raw HTTP bodies, process argv, business graphs or environment data.
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__,
                          "error": str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "operation_failed"}),
              file=sys.stderr)
        raise SystemExit(1)
