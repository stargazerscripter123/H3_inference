#!/usr/bin/env python3
"""Read-only Linux resource sampling for this Omni deployment (stdlib only).

Device memory is nvidia-smi's whole-device usage, including display and other
processes. It is NOT PyTorch allocated/reserved memory. Process RSS is filtered
to this deployment, and summed RSS can double-count shared pages.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


GPU_FIELDS = "index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total"
GPU_SCOPE = "whole-device nvidia-smi usage; includes display and unrelated processes; not PyTorch allocator"
RSS_SCOPE = "same-user project executable/script paths and identified RayWorkerWrapper actors; summed RSS may double-count shared pages"


def positive_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def positive_pid(value: str) -> int:
    pid = int(value)
    if pid <= 0:
        raise argparse.ArgumentTypeError("PID must be positive")
    return pid


def parse_gpu_csv(text: str) -> list[dict]:
    output = []
    seen = set()
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 7:
            raise ValueError("unexpected nvidia-smi column count")
        index = int(row[0])
        if index < 0 or index in seen:
            raise ValueError("invalid or duplicated GPU index")
        seen.add(index)
        values = []
        for value in row[3:]:
            value = value.strip()
            if value in {"N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
                values.append(None)
            else:
                number = float(value)
                if not math.isfinite(number) or number < 0:
                    raise ValueError("invalid GPU metric")
                values.append(number)
        if any(value is not None and value > 100 for value in values[:2]):
            raise ValueError("invalid utilization percentage")
        if not row[1].strip():
            raise ValueError("missing GPU UUID")
        output.append(dict(index=index, uuid=row[1].strip(), name=row[2].strip(),
                           utilization_gpu_percent=values[0], utilization_memory_percent=values[1],
                           memory_used_mib=values[2], memory_total_mib=values[3]))
    if not output:
        raise ValueError("empty nvidia-smi response")
    return output


def parse_meminfo(text: str) -> dict:
    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    values = {}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if separator and key in wanted:
            fields = rest.split()
            if len(fields) != 2 or fields[1] != "kB" or int(fields[0]) < 0:
                raise ValueError("invalid /proc/meminfo metric")
            values[key] = int(fields[0]) * 1024
    if set(values) != wanted:
        raise ValueError("missing /proc/meminfo fields")
    if values["MemAvailable"] > values["MemTotal"] or values["SwapFree"] > values["SwapTotal"]:
        raise ValueError("inconsistent /proc/meminfo totals")
    return dict(mem_total_bytes=values["MemTotal"], mem_available_bytes=values["MemAvailable"],
                swap_total_bytes=values["SwapTotal"], swap_free_bytes=values["SwapFree"])


def parse_proc_stat(text: str) -> dict:
    # comm can contain spaces and parentheses; numeric fields follow the last ).
    fields = text.rsplit(")", 1)[1].split()
    return {"state": fields[0], "ppid": int(fields[1]), "start_ticks": int(fields[19])}


def read_arg(stream) -> str:
    """Read one NUL-delimited token only; never read subsequent script arguments."""
    value = bytearray()
    for _ in range(4096):
        byte = stream.read(1)
        if byte in (b"", b"\0"):
            return value.decode("utf-8", errors="replace")
        value.extend(byte)
    return ""  # Unusually long process titles are not used for discovery.


def within(value: str, root: Path) -> bool:
    return value.startswith("/") and Path(os.path.normpath(value)).is_relative_to(root)


def process_scope(proc: Path, project_root: Path) -> str | None:
    """Inspect executable/script tokens; never environ, code after -c, or app args."""
    with (proc / "cmdline").open("rb", buffering=0) as stream:
        executable = read_arg(stream)
        if within(executable, project_root / "envs/omni"):
            return "project_venv"
        if within(executable, project_root / "omni-deployment"):
            return "project_executable"
        if executable.startswith("ray::RayWorkerWrapper"):
            return "ray_actor_candidate"
        name = Path(executable).name
        if not (name.startswith("python") or name in {"sh", "bash", "zsh"}):
            return None
        # Known switches have no following values. Unknown switches (including
        # -c and -m) stop scanning, so code/module/app arguments are not read.
        for _ in range(8):
            token = read_arg(stream)
            if token in {"-u", "-B", "-E", "-I", "-O", "-OO", "-s", "-S", "-q", "--"}:
                continue
            if not token or token.startswith("-"):
                return None
            if not token.startswith("/"):
                token = os.path.join(os.readlink(proc / "cwd"), token)
            return "project_script" if within(token, project_root / "omni-deployment") else None
    return None


def process_snapshot(proc_root: Path, project_root: Path, actor_identities: dict[int, int]) -> tuple[list[dict], dict]:
    candidates = {}
    skipped = 0
    unavailable = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            info = parse_proc_stat((directory / "stat").read_text())
            if info["state"] == "Z":
                continue
            info["scope"] = process_scope(directory, project_root)
            candidates[int(directory.name)] = info
        except (OSError, ValueError, IndexError):
            skipped += 1  # Exit races and denied reads are expected; no cmdline is logged.
    selected = {pid for pid, info in candidates.items()
                if info["scope"] and info["scope"] != "ray_actor_candidate"}
    rows = []
    for pid, info in candidates.items():
        scope = info["scope"]
        ancestor = None
        if scope == "ray_actor_candidate":
            if actor_identities.get(pid) == info["start_ticks"]:
                scope = "explicit_ray_actor"
            else:
                parent = info["ppid"]
                visited = {pid}
                while parent in candidates and parent not in visited:
                    if parent in selected:
                        ancestor = parent
                        break
                    visited.add(parent)
                    parent = candidates[parent]["ppid"]
                if ancestor is None:
                    continue  # A matching Ray title alone is not ownership evidence.
                scope = "ray_actor_project_ancestor"
        elif pid not in selected:
            continue
        try:
            directory = proc_root / str(pid)
            rss = None
            for line in (directory / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    if len(fields) != 3 or fields[2] != "kB" or int(fields[1]) < 0:
                        raise ValueError("invalid RSS")
                    rss = int(fields[1]) * 1024
                    break
            if rss is None:
                raise ValueError("RSS unavailable")
            current = parse_proc_stat((directory / "stat").read_text())
            if current["start_ticks"] != info["start_ticks"]:
                raise ValueError("PID reused during sample")
            rows.append(dict(pid=pid, start_ticks=info["start_ticks"], rss_bytes=rss,
                             scope=scope, project_ancestor_pid=ancestor))
        except (OSError, ValueError, IndexError) as error:
            skipped += 1
            unavailable.append({"pid": pid, "start_ticks": info["start_ticks"], "error_type": type(error).__name__})
    return sorted(rows, key=lambda row: row["pid"]), {
        "skipped_or_raced_processes": skipped, "selected_processes_without_rss": unavailable}


def sample(args, actors: dict[int, int]) -> dict:
    started = time.monotonic()
    result = {"kind": "sample", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "gpus": [], "host": None, "processes": [], "errors": []}
    try:
        response = subprocess.run(
            [args.nvidia_smi, "--query-gpu=" + GPU_FIELDS, "--format=csv,noheader,nounits", "-i", "0,1"],
            capture_output=True, text=True, timeout=5, check=True)
        result["gpus"] = parse_gpu_csv(response.stdout)
        if {row["index"] for row in result["gpus"]} != {0, 1}:
            result["errors"].append({"source": "gpu", "error": "expected physical GPUs 0 and 1"})
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["errors"].append({"source": "gpu", "error_type": type(error).__name__})
    try:
        result["host"] = parse_meminfo(Path("/proc/meminfo").read_text())
    except (OSError, ValueError) as error:
        result["errors"].append({"source": "host", "error_type": type(error).__name__})
    try:
        result["processes"], result["process_discovery"] = process_snapshot(Path("/proc"), args.project_root, actors)
        if result["process_discovery"]["selected_processes_without_rss"]:
            result["errors"].append({"source": "processes", "error": "selected process RSS unavailable; sum is partial"})
    except OSError as error:
        result["errors"].append({"source": "processes", "error_type": type(error).__name__})
    result["selected_process_rss_sum_bytes"] = (sum(row["rss_bytes"] for row in result["processes"])
                                                if "process_discovery" in result else None)
    result["sampling_duration_s"] = time.monotonic() - started
    return result


def update_extreme(mapping: dict, key: str, value, *, minimum: bool = False) -> None:
    if value is not None:
        mapping[key] = value if mapping.get(key) is None else (min if minimum else max)(mapping[key], value)


class Peaks:
    def __init__(self):
        self.value = {"samples": 0, "gpu_samples": 0, "host_samples": 0, "error_counts": {},
                      "gpus": {}, "host": {}, "processes": {}, "peak_selected_process_rss_sum_bytes": None,
                      "samples_without_selected_processes": 0, "max_skipped_or_raced_processes": 0,
                      "max_sampling_duration_s": 0.0, "gpu_memory_scope": GPU_SCOPE, "process_rss_scope": RSS_SCOPE}

    def add(self, row: dict) -> None:
        result = self.value
        result["samples"] += 1
        result["gpu_samples"] += bool(row["gpus"])
        result["host_samples"] += row["host"] is not None
        result["samples_without_selected_processes"] += not bool(row["processes"])
        update_extreme(result, "max_skipped_or_raced_processes", row.get("process_discovery", {}).get("skipped_or_raced_processes"))
        update_extreme(result, "max_sampling_duration_s", row["sampling_duration_s"])
        update_extreme(result, "peak_selected_process_rss_sum_bytes", row["selected_process_rss_sum_bytes"])
        for error in row["errors"]:
            source = error["source"]
            result["error_counts"][source] = result["error_counts"].get(source, 0) + 1
        for gpu in row["gpus"]:
            entry = result["gpus"].setdefault(gpu["uuid"], {"index": gpu["index"], "name": gpu["name"], "samples": 0})
            entry["samples"] += 1
            for field in ("utilization_gpu_percent", "utilization_memory_percent", "memory_used_mib"):
                update_extreme(entry, "peak_" + field, gpu[field])
            entry["memory_total_mib"] = gpu["memory_total_mib"]
        if row["host"] is not None:
            host = row["host"]
            update_extreme(result["host"], "min_mem_available_bytes", host["mem_available_bytes"], minimum=True)
            update_extreme(result["host"], "min_swap_free_bytes", host["swap_free_bytes"], minimum=True)
            update_extreme(result["host"], "peak_mem_total_minus_available_bytes", host["mem_total_bytes"] - host["mem_available_bytes"])
            update_extreme(result["host"], "peak_swap_used_bytes", host["swap_total_bytes"] - host["swap_free_bytes"])
        for proc in row["processes"]:
            key = f"{proc['pid']}:{proc['start_ticks']}"
            entry = result["processes"].setdefault(key, {**proc, "samples": 0})
            entry.pop("rss_bytes", None)
            entry["samples"] += 1
            update_extreme(entry, "peak_rss_bytes", proc["rss_bytes"])


class Stop:
    def __init__(self):
        self.event = threading.Event()
        self.signal_number = None

    def __call__(self, signum, _frame=None):
        self.signal_number = signum
        self.event.set()


def monitor(args, actors: dict[int, int], stop: Stop, sampler=sample) -> dict:
    summary_path = args.output.with_suffix(".summary.json")
    if args.output == summary_path:
        raise ValueError("--output must not end in .summary.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    peaks = Peaks()
    started = time.monotonic()
    fatal = None
    # Exclusive creation prevents overwriting evidence from a previous run.
    with args.output.open("x", encoding="utf-8") as output, summary_path.open("x", encoding="utf-8") as summary:
        def emit(row):
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
        emit({"kind": "metadata", "schema_version": 1, "monitor_pid": os.getpid(),
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "project_root": str(args.project_root),
              "interval_s": args.interval, "duration_s": args.duration, "physical_gpu_indices": [0, 1],
              "explicit_actor_identities": actors, "gpu_memory_scope": GPU_SCOPE, "process_rss_scope": RSS_SCOPE,
              "privacy": "no environ, credentials, full command lines, or script arguments are read or recorded",
              "peak_note": "observed samples only; short peaks between samples may be missed"})
        try:
            while not stop.event.is_set():
                tick = time.monotonic()
                if args.duration is not None and tick - started >= args.duration:
                    break
                row = sampler(args, actors)
                row["elapsed_s"] = time.monotonic() - started
                peaks.add(row)
                emit(row)
                delay = max(0.0, args.interval - (time.monotonic() - tick))
                if args.duration is not None:
                    delay = min(delay, max(0.0, args.duration - (time.monotonic() - started)))
                stop.event.wait(delay)
        except Exception as error:
            fatal = type(error).__name__
            raise
        finally:
            result = {"kind": "summary", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_s": time.monotonic() - started, "stop_signal": stop.signal_number,
                      "stop_reason": "sampling_error" if fatal else ("signal" if stop.signal_number else "duration"),
                      "fatal_error_type": fatal, **peaks.value}
            emit(result)
            summary.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            summary.flush()
            os.fsync(summary.fileno())
            os.fsync(output.fileno())
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-root", type=Path, default=Path("/home/isaac/workdir/h3"))
    result.add_argument("--interval", type=positive_number, default=1.0, help="seconds between sample starts (minimum 0.1)")
    result.add_argument("--duration", type=positive_number, help="seconds; omitted means run until SIGINT/SIGTERM")
    result.add_argument("--output", type=Path, required=True, help="new JSONL file; sibling .summary.json is also created")
    result.add_argument("--nvidia-smi", default="nvidia-smi")
    result.add_argument("--ray-actor-pid", type=positive_pid, action="append", default=[],
                        help="already identified same-user RayWorkerWrapper PID; repeatable, bound to its start time")
    return result


def main() -> int:
    p = parser()
    args = p.parse_args()
    if sys.platform != "linux":
        p.error("Linux /proc is required; use parser/fixture tests on other systems")
    if args.interval < 0.1:
        p.error("--interval must be at least 0.1 seconds")
    args.project_root = args.project_root.expanduser()
    if not args.project_root.is_absolute():
        p.error("--project-root must be absolute")
    args.project_root = Path(os.path.normpath(args.project_root))
    args.output = args.output.expanduser().absolute()
    actors = {}
    try:
        for pid in args.ray_actor_pid:
            proc = Path("/proc") / str(pid)
            if proc.stat().st_uid != os.getuid() or process_scope(proc, args.project_root) != "ray_actor_candidate":
                raise ValueError(f"PID {pid} is not a same-user RayWorkerWrapper actor")
            actors[pid] = parse_proc_stat((proc / "stat").read_text())["start_ticks"]
        stop = Stop()
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        result = monitor(args, actors, stop)
        print(json.dumps({"output": str(args.output), "summary": str(args.output.with_suffix('.summary.json')),
                          "samples": result["samples"], "sampling_error_counts": result["error_counts"]}))
        return 0 if result["samples"] and not result["error_counts"] else 2
    except (OSError, ValueError, IndexError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
