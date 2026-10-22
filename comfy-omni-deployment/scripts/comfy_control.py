#!/usr/bin/env python3
"""Own isolated single-GPU ComfyUI Omni services in explicit fixed slots on Linux.

No global process cleanup is performed. Stop always requires the owner token,
UID, process group, start ticks and wrapper argv to match the saved service.
READY means the HTTP server and native reference node are ready; models are
loaded lazily by the first submitted graph. Model SHA checks belong to the
comparison client; this controller checks pinned model names and file sizes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
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
INSTANCES = {"default": (None, PORT), "gpu0": (0, 8189), "gpu1": (1, 8190), "sp2": (0, 8191)}
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


def instance_settings(instance: str, gpu: int | None = None) -> tuple[int, int]:
    if instance not in INSTANCES:
        raise ValueError("unknown_instance_slot")
    fixed_gpu, port = INSTANCES[instance]
    if gpu is None:
        gpu = 1 if fixed_gpu is None else fixed_gpu
    if type(gpu) is not int or gpu not in (0, 1):
        raise ValueError("gpu_must_be_single_physical_device_0_or_1")
    if fixed_gpu is not None and gpu != fixed_gpu:
        raise ValueError("instance_gpu_mismatch")
    return gpu, port


def instance_run(root: Path, instance: str) -> Path:
    instance_settings(instance)
    base = root / "run/comfy-omni"
    path = base if instance == "default" else base / "instances" / instance
    if path != path.resolve():
        raise ValueError("instance_run_directory_must_not_be_aliased")
    return confined(path, (root,), allow_root=False)


def configuration(root: Path, gpu: int | None = None, input_dir: str | None = None,
                  output_dir: str | None = None, instance: str = "default", *,
                  sp_verify: bool = False, sp_profile: bool = False) -> dict:
    gpu, port = instance_settings(instance, gpu)
    if type(sp_verify) is not bool or type(sp_profile) is not bool or ((sp_verify or sp_profile) and instance != "sp2"):
        raise ValueError("sp_diagnostics_require_sp2_slot")
    root = project_root(root)
    run = instance_run(root, instance)
    # Explicit slots write only their own input and output trees. The legacy
    # default keeps its existing paths and fingerprint-compatible config shape.
    input_base = root / "inputs" if instance == "default" else run / "input"
    input_roots = tuple(confined(path, (root,), allow_root=False)
                        for path in (input_base, run / "input"))
    output_base = root / "outputs/comfy-omni"
    if instance != "default":
        output_base = output_base / "instances" / instance
        if output_base != output_base.resolve() or run / "input" != (run / "input").resolve():
            raise ValueError("instance_io_directory_must_not_be_aliased")
    output_base = confined(output_base, (root,), allow_root=False)
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
              "listen": "127.0.0.1", "port": port, "gpu": gpu,
              "input_dir": input_dir, "output_dir": output_dir,
              "models_dir": str(models), "run_dir": str(run),
              "user_dir": str(confined(run / "user", (run,), allow_root=False)),
              "temp_base": str(confined(run / "temp", (run,), allow_root=False)),
              "disable_metadata": True, "disable_all_custom_nodes": True,
              "disable_auto_launch": True, "single_gpu": True,
              "readiness_scope": READINESS,
              "pins_sha256": sha256(HERE.parent / "pins.json"),
              "wrapper_sha256": sha256(HERE / "comfy_service.py")}
    if instance != "default":
        config["instance"] = instance
    if instance == "sp2":
        config.update(single_gpu=False, gpu_ids=[0, 1], parallelism="ulysses-sp2",
                      sp_verify=sp_verify, sp_profile=sp_profile,
                      sp_contract_sha256=sha256(HERE.parent / "sp-node-contract.json"))
    if instance == "sp2" and sp_verify:
        # Native DynamicVRAM headroom only for the first-forward diagnostic.
        # ComfyUI converts this GiB value using 1024**3. Each diagnostic keeps
        # its independent physical-free guard; this setting does not relax it.
        config["diagnostic_vram_headroom_gib"] = 4.0
    return config


def validate_config(config: dict) -> None:
    root = project_root(config["project_root"])
    expected = configuration(root, config["gpu"], config["input_dir"], config["output_dir"],
                             config.get("instance", "default"),
                             sp_verify=config.get("sp_verify", False), sp_profile=config.get("sp_profile", False))
    headroom_key = "diagnostic_vram_headroom_gib"
    if ((headroom_key in config) != (headroom_key in expected)
            or any(config.get(key) != value for key, value in expected.items())):
        raise ValueError("launch_configuration_mismatch")
    token = config.get("owner_token")
    if token is not None:
        if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
            raise ValueError("invalid_owner_token")
        if config.get("temp_dir") != str(Path(config["temp_base"]) / token):
            raise ValueError("temp_directory_mismatch")


def command_arguments(config: dict) -> list[str]:
    """Only the SP2 slot enables the fixed, hash-verified custom node."""
    arguments = [str(Path(config["source"]) / "main.py"),
            "--listen", "127.0.0.1", "--port", str(config["port"]),
            "--disable-metadata", "--disable-auto-launch", "--disable-all-custom-nodes",
            "--preview-method", "none", "--input-directory", config["input_dir"],
            "--output-directory", config["output_dir"],
            "--temp-directory", config["temp_dir"], "--user-directory", config["user_dir"],
            "--models-directory", config["models_dir"]]
    if config.get("instance") == "sp2":
        arguments += ["--base-directory", str(HERE.parent),
                      "--whitelist-custom-nodes", "minimax_sp_pdd"]
        if config.get("sp_verify") is True:
            arguments += ["--vram-headroom", str(config["diagnostic_vram_headroom_gib"])]
    return arguments


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
    if config.get("instance") == "sp2":
        env.update(CUDA_VISIBLE_DEVICES="0,1", MINIMAX_SP_DEVICES="0,1",
                   MINIMAX_SP_STARTUP_TIMEOUT="600", MINIMAX_SP_COLLECTIVE_TIMEOUT="180",
                   MINIMAX_SP_LOGDIR=config["temp_dir"])
        if config.get("sp_verify"):
            env["MINIMAX_SP_VERIFY"] = "1"
        if config.get("sp_profile"):
            env["MINIMAX_SP_PROFILE"] = "1"
    return env


def verify_sp_node() -> dict:
    contract = json.loads((HERE.parent / "sp-node-contract.json").read_text())
    base = HERE.parent / "custom_nodes/minimax_sp_pdd"
    if base != base.resolve():
        raise RuntimeError("sp_node_alias_rejected")
    expected = contract["files"]
    actual = {str(p.relative_to(base)) for p in base.rglob("*.py")}
    if actual != {n for n in expected if n.endswith(".py")}:
        raise RuntimeError("sp_node_python_inventory_mismatch")
    for name, digest in expected.items():
        path = base / name
        if path != path.resolve() or not path.is_file() or sha256(path) != digest:
            raise RuntimeError("sp_node_source_hash_mismatch")
    return {"upstream_commit": contract["upstream_commit"],
            "contract_sha256": sha256(HERE.parent / "sp-node-contract.json")}


def verify_installation(config: dict) -> dict:
    validate_config(config)
    if config.get("instance") == "sp2":
        verify_sp_node()
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


@contextmanager
def locked_instance_starts(root: Path):
    """Serialize reservation checks across slots, without locking long inference."""
    run = instance_run(root, "default")
    run.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(run / "instance-start.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def state_port(state: dict) -> int:
    instance = state.get("instance", "default")
    _, expected = instance_settings(instance, state.get("gpu"))
    if type(state.get("port")) is not int or state["port"] != expected:
        raise RuntimeError("saved_instance_port_mismatch")
    return expected


def validate_state_scope(state: dict, run: Path) -> None:
    """A receipt from another slot cannot authorize status, retirement or stop."""
    instance = state.get("instance", "default")
    state_port(state)
    if instance == "default":
        expected_suffix = ("run", "comfy-omni")
    else:
        expected_suffix = ("run", "comfy-omni", "instances", instance)
        if state.get("gpu") != INSTANCES[instance][0]:
            raise RuntimeError("saved_instance_gpu_mismatch")
    config = Path(state.get("config", ""))
    if (not run.is_absolute() or run != run.resolve() or tuple(run.parts[-len(expected_suffix):]) != expected_suffix
            or not config.is_absolute() or config.parent != run or config != config.resolve()
            or config.suffix != ".json"):
        raise RuntimeError("saved_instance_scope_mismatch")


def require_available_slot(config: dict, run: Path) -> None:
    """Known other slots may coexist only on a different, verified GPU.

    This is not a machine-wide idle probe. The runner still checks FastVideo,
    unknown GPU processes, ports and host RAM before authorizing the pair.
    The CLI holds the shared start lock while this check and launch execute.
    """
    for instance in INSTANCES:
        other_run = instance_run(Path(config["project_root"]), instance)
        if other_run == run:
            continue
        other = read_state(other_run)
        if other is None:
            continue
        if other.get("identity_established") is False:
            if unverified_alive(other):
                raise RuntimeError("other_instance_identity_unverified_no_signals_sent")
            continue
        live = owned(other)
        if not live and group_members(other["pgid"]):
            raise RuntimeError("other_instance_orphaned_no_signals_sent")
        reserved = {0, 1} if other.get("instance") == "sp2" else {other.get("gpu")}
        requested = {0, 1} if config.get("instance") == "sp2" else {config["gpu"]}
        if live and (type(other.get("gpu")) is not int or reserved.intersection(requested)):
            raise RuntimeError("gpu_reserved_by_other_instance_no_signals_sent")


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
    address = f"0100007F:{state_port(state):04X}"
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
    base = f"http://127.0.0.1:{state_port(state)}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(base + "/system_stats", timeout=3) as response:
            stats = json.load(response)
        if stats.get("system", {}).get("comfyui_version") != VERSION:
            return {"status": "IDENTITY_MISMATCH"}
        devices = stats.get("devices", [])
        expected_devices = [0, 1] if state.get("instance") == "sp2" else [0]
        if ([d.get("index") for d in devices] != expected_devices
                or any(d.get("type") != "cuda" for d in devices)):
            return {"status": "IDENTITY_MISMATCH"}
        with opener.open(base + "/object_info/MiniMaxH3ReferenceToVideo", timeout=3) as response:
            nodes = json.load(response)
        if "MiniMaxH3ReferenceToVideo" not in nodes:
            return {"status": "NATIVE_NODE_MISSING"}
        if state.get("instance") == "sp2":
            with opener.open(base + "/object_info/MiniMaxH3SPUNETLoader", timeout=3) as response:
                sp_nodes = json.load(response)
            if "MiniMaxH3SPUNETLoader" not in sp_nodes:
                return {"status": "NATIVE_NODE_MISSING"}
        return {"status": "READY", "readiness_scope": READINESS}
    except (OSError, ValueError, urllib.error.URLError):
        return {"status": "STARTING_OR_UNHEALTHY", "readiness_scope": READINESS}


def read_state(run: Path) -> dict | None:
    path = run / "service.json"
    if not path.exists():
        return None
    if path.is_symlink():
        raise RuntimeError("saved_state_alias_rejected")
    state = json.loads(path.read_text())
    validate_state_scope(state, run)
    return state


def check_port(port: int = PORT) -> None:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def retired(run: Path, state: dict) -> None:
    state = {**state, "stopped_at": time.time()}
    token = state["owner_token"]
    path = run / f"stopped-{token}-{time.time_ns()}.json"
    atomic_json(path, state)
    (run / "service.json").unlink(missing_ok=True)


def start(config: dict, run: Path) -> dict:
    if Path(config["run_dir"]) != run:
        raise ValueError("launch_instance_run_mismatch")
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
    require_available_slot(config, run)
    check_port(config["port"])
    assets = verify_installation(config)
    token = uuid.uuid4().hex
    launch = {**config, "owner_token": token,
              "temp_dir": str(Path(config["temp_base"]) / token)}
    root = Path(config["project_root"])
    for key in ("input_dir", "output_dir", "temp_dir", "user_dir"):
        Path(launch[key]).mkdir(parents=True, exist_ok=True, mode=0o700)
    log_base = root / "logs/comfy-omni"
    if config.get("instance", "default") != "default":
        log_base = log_base / "instances" / config["instance"]
        if log_base != log_base.resolve():
            raise ValueError("instance_log_directory_must_not_be_aliased")
    log_dir = confined(log_base, (root,), allow_root=False)
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
        process = subprocess.Popen(argv, cwd=config["source"], env=child_environment(launch),
                                   stdin=null, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    state = {"schema_version": 1, "pid": process.pid, "pgid": process.pid, "uid": os.getuid(),
             "owner_token": token, "service_script": script, "config": str(config_path),
             "fingerprint": digest, "comfy_commit": PIN, "comfy_version": VERSION,
             "gpu": config["gpu"], "port": config["port"], "input_dir": config["input_dir"],
             "output_dir": config["output_dir"], "models_dir": config["models_dir"],
             "source": config["source"], "python": config["python"],
             "disable_metadata": True, "disable_all_custom_nodes": True,
             "started_at": time.time(), "log": str(log_path), "asset_preflight": assets,
             "readiness_scope": READINESS}
    if "instance" in config:
        state["instance"] = config["instance"]
    if config.get("instance") == "sp2":
        state.update(gpu_ids=config["gpu_ids"], parallelism=config["parallelism"],
                     sp_contract_sha256=config["sp_contract_sha256"],
                     sp_verify=config["sp_verify"], sp_profile=config["sp_profile"])
        if "diagnostic_vram_headroom_gib" in config:
            state["diagnostic_vram_headroom_gib"] = config["diagnostic_vram_headroom_gib"]
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
            "started_at", "stop_requested_at", "force_requested_at", "readiness_scope", "reused", "instance", "gpu_ids", "parallelism", "sp_contract_sha256", "sp_verify", "sp_profile", "diagnostic_vram_headroom_gib"}
    return {key: value for key, value in result.items() if key in keys}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "validate", "start", "status", "stop"))
    parser.add_argument("--project-root", default="/home/isaac/workdir/h3")
    parser.add_argument("--instance", choices=tuple(INSTANCES), default="default",
                        help="default: legacy 8188; gpu0: GPU0/8189; gpu1: GPU1/8190, each with isolated state and I/O")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=None,
                        help="legacy default selects GPU1; explicit slots enforce their fixed GPU")
    parser.add_argument("--sp-verify", action="store_true", help="SP2 first-forward numerical comparison; diagnostic only")
    parser.add_argument("--sp-profile", action="store_true", help="SP2 per-forward synchronization; diagnostic only")
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
    instance_settings(args.instance, args.gpu)
    run = instance_run(root, args.instance)
    if args.action == "render":
        result = configuration(root, args.gpu, args.input_dir, args.output_dir, args.instance,
                               sp_verify=args.sp_verify, sp_profile=args.sp_profile)
    else:
        if sys.platform != "linux":
            raise RuntimeError("service_lifecycle_requires_linux_proc")
        with (locked_instance_starts(root) if args.action == "start" else nullcontext()), locked(run):
            if args.action in ("start", "validate"):
                config = configuration(root, args.gpu, args.input_dir, args.output_dir, args.instance,
                               sp_verify=args.sp_verify, sp_profile=args.sp_profile)
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
