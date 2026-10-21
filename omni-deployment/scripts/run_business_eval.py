#!/usr/bin/env python3
"""Run a sealed private business protocol with a fresh owned SP2 service per job.

Source env.popos-5090.sh before invoking. This runner never retries generation,
rewrites prompts, kills a service forcibly, or copies code to another machine.
Records and client logs are private runtime artifacts, not public reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
READY_TIMEOUT = 600.0
STOP_TIMEOUT = 600.0
STATUS_INTERVAL = 5.0
STOP_INTERVAL = 30.0
CLIENT_TIMEOUT = 1800
EXPECTED = {"width": 480, "height": 832, "num_frames": 362, "fps": 24,
            "num_inference_steps": 8, "guidance_scale": 1, "vae_offload": True,
            "expandable_segments": True, "sp_profile": False, "profile": "sp2"}
POLICIES = {"cap480": ("cap", 480), "cap720": ("cap", 720), "original": ("original", None)}


def now() -> float:
    return time.time()


def sha_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def checked_sha(value) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid_sha256")
    return value


def within(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    path = (root / path if not path.is_absolute() else path).resolve()
    require(path.is_relative_to(root), "path_outside_project")
    return path


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "json_object_required")
    return value


def validate_protocol(path: Path, root: Path) -> tuple[dict, dict[str, dict]]:
    protocol = read_object(path)
    require(protocol.get("schema_version") == 1, "unsupported_protocol_schema")
    require(isinstance(protocol.get("tag"), str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", protocol["tag"]) is not None, "invalid_tag")
    expected = protocol.get("expected", {})
    require(all(expected.get(k) == v and (type(expected.get(k)) is bool if type(v) is bool else True)
                for k, v in EXPECTED.items()), "unexpected_runtime_recipe")
    cases = protocol.get("cases")
    require(isinstance(cases, list) and len(cases) == 3, "exactly_three_cases_required")
    case_map = {c.get("case_id"): c for c in cases if isinstance(c, dict)}
    require(set(case_map) == {"case-01", "case-02", "case-03"}, "anonymous_case_ids_required")
    for case in case_map.values():
        checked_sha(case.get("prompt_sha256"))
        refs = case.get("references")
        require(isinstance(refs, list) and 1 <= len(refs) <= 8, "invalid_case_references")
        for ref in refs:
            require(isinstance(ref, dict), "invalid_reference")
            checked_sha(ref.get("sha256"))
            require(all(type(ref.get(k)) is int and ref[k] > 0 for k in ("width", "height")), "invalid_reference_dimensions")
            source = within(ref["remote_path"], root)
            require(source.is_file(), "reference_file_missing")
            require(sha_file(source) == ref["sha256"], "reference_sha256_mismatch")
    policies = protocol.get("policies")
    require(isinstance(policies, list) and len(policies) == 3, "exactly_three_policies_required")
    policy_map = {p.get("policy_id"): p for p in policies if isinstance(p, dict)}
    require(set(policy_map) == set(POLICIES), "invalid_policy_ids")
    for key, values in POLICIES.items():
        require((policy_map[key].get("resize_policy"), policy_map[key].get("max_short_edge")) == values, "invalid_resize_policy")
    jobs = protocol.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 9, "exactly_nine_jobs_required")
    ids, pairs, prepared, paired_plans = set(), set(), {}, {}
    for job in jobs:
        require(isinstance(job, dict), "invalid_job")
        jid = job.get("job_id")
        require(isinstance(jid, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", jid) is not None and jid not in ids, "invalid_or_duplicate_job_id")
        ids.add(jid)
        pair = (job.get("case_id"), job.get("policy_id"))
        require(pair[0] in case_map and pair[1] in policy_map and pair not in pairs, "invalid_or_duplicate_job_pair")
        pairs.add(pair)
        require(job.get("profile") == "sp2", "job_profile_must_be_sp2")
        require((job.get("resize_policy"), job.get("max_short_edge")) == POLICIES[pair[1]], "job_policy_mismatch")
        plan_path = within(job["plan"], root)
        require(sha_file(plan_path) == checked_sha(job.get("plan_sha256")), "plan_sha256_mismatch")
        plan = read_object(plan_path)
        requests = plan.get("requests")
        require(isinstance(requests, list) and len(requests) >= 2, "warmup_and_measured_requests_required")
        case = case_map[pair[0]]
        signature, measured = [], False
        for item in requests:
            require(isinstance(item, dict) and item.get("case_id") == pair[0], "plan_case_mismatch")
            prompt = item.get("prompt")
            require(isinstance(prompt, str) and hashlib.sha256(prompt.encode()).hexdigest() == case["prompt_sha256"], "plan_prompt_sha256_mismatch")
            require((item.get("width"), item.get("height"), item.get("frames")) == (480, 832, 362), "plan_output_geometry_mismatch")
            require(type(item.get("seed")) is int and -(2**63) <= item["seed"] < 2**63, "invalid_seed")
            phase = item.get("phase")
            require(phase in {"warmup", "measured"} and not (measured and phase == "warmup"), "invalid_phase_order")
            measured |= phase == "measured"
            refs = item.get("references")
            require(isinstance(refs, list) and len(refs) == len(case["references"]), "plan_reference_count_mismatch")
            for actual, wanted in zip(refs, case["references"], strict=True):
                require(isinstance(actual, dict) and actual.get("type") == "image" and
                        within(actual["source"], root) == within(wanted["remote_path"], root), "plan_reference_order_mismatch")
            condition = item.get("cache_condition")
            require(condition in {"cold", "warm-new", "warm-fixed-ref", "warm-full-conditioning", "unspecified"}, "invalid_cache_condition")
            signature.append((phase, item["seed"], condition))
        require(requests[0]["phase"] == "warmup" and measured, "warmup_and_measured_requests_required")
        if pair[0] in paired_plans:
            require(signature == paired_plans[pair[0]], "cross_policy_seed_or_phase_mismatch")
        paired_plans[pair[0]] = signature
        prepared[jid] = {"plan": plan_path, "request_count": len(requests), "case": case}
    return protocol, prepared


def last_json(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text, index)
            if isinstance(value, dict) and not text[end:].strip():
                return value
        except ValueError:
            pass
    return None


def process_identity(pid: int) -> dict:
    proc = Path(f"/proc/{pid}")
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "pgid": int(fields[2]), "uid": proc.stat().st_uid, "start_ticks": fields[19]}


class Runner:
    def __init__(self, root: Path, record_path: Path, output: Path, protocol_path: Path, protocol: dict, prepared: dict):
        self.root, self.record_path, self.output = root, record_path, output
        self.protocol, self.prepared = protocol, prepared
        self.interrupted = False
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid_git_commit")
        self.record = {"schema_version": 1, "status": "running", "protocol_path": str(protocol_path),
                       "protocol_sha256": sha_file(protocol_path), "git_commit": commit, "started_at": now(),
                       "runner": process_identity(os.getpid()), "jobs": [dict(job, status="not_run", started_at=None,
                       ready_at=None, finished_at=None, report=None, output_root=str(output / job["job_id"]),
                       service=None, service_final_state=None, service_final_detail=None,
                       cleanup_errors=[]) for job in protocol["jobs"]]}

    def save(self) -> None:
        temporary = self.record_path.with_name(self.record_path.name + ".tmp-" + uuid.uuid4().hex)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(self.record, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.record_path)

    def environment(self, job: dict) -> dict:
        env = dict(os.environ)
        env.update(H3_OMNI_ROOT=str(self.root), OMNI_VAE_OFFLOAD="1", OMNI_EXPANDABLE_SEGMENTS="1",
                   OMNI_SP_PROFILE="0", OMNI_GPU_IDS="0,1", OMNI_REFERENCE_IMAGE_RESIZE_POLICY=job["resize_policy"],
                   OMNI_REFERENCE_IMAGE_MAX_SHORT_EDGE=str(job["max_short_edge"] or 0),
                   OMNI_OUTPUT_DIR=str(Path(job["output_root"]) / "server"))
        return env

    def control(self, action: str, env: dict, *extra: str, timeout: float = 20) -> tuple[int, dict | None, str]:
        result = subprocess.run([sys.executable, str(HERE / "omni_control.py"), action, "sp2", *extra],
                                env=env, cwd=self.root, capture_output=True, text=True, timeout=timeout, check=False)
        diagnostic = hashlib.sha256((result.stdout + result.stderr).encode()).hexdigest()
        return result.returncode, last_json(result.stdout), diagnostic

    def status(self, env: dict, *, timeout: float = 20) -> dict:
        rc, value, _ = self.control("status", env, timeout=timeout)
        require(rc == 0 and value is not None, "controller_status_failed")
        return value

    @staticmethod
    def final_state(job: dict, value: dict) -> None:
        job["service_final_state"] = value.get("status", "UNKNOWN")
        job["service_final_detail"] = value

    def stop_service(self, job: dict, env: dict, token: str) -> None:
        deadline = time.monotonic() + STOP_TIMEOUT
        try:
            rc, value, diagnostic = self.control("stop", env, "--expected-owner-token", token,
                                                 "--stop-timeout", "30", timeout=45)
            job["stop_attempt"] = {"returncode": rc, "diagnostic_sha256": diagnostic}
            if rc == 0 and value and value.get("status") == "STOPPED":
                self.final_state(job, value)
                return
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            job["stop_attempt"] = {"error_type": type(error).__name__}
        # The one SIGTERM may need time to drain workers. Do not send force or
        # launch another generation request while graceful shutdown is pending.
        while time.monotonic() < deadline:
            try:
                state = self.status(env, timeout=min(20, max(0.1, deadline - time.monotonic())))
                self.final_state(job, state)
                self.save()
                if state.get("status") == "STOPPED":
                    return
                if state.get("owner_token") != token:
                    job["cleanup_errors"].append("service_owner_changed_refused_signals")
                    return
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            time.sleep(min(STOP_INTERVAL, max(0, deadline - time.monotonic())))
        job["cleanup_errors"].append("service_stop_deadline_not_confirmed")

    @staticmethod
    def stop_child(process: subprocess.Popen | None, label: str, job: dict) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            # Only the child process group created by this Popen is signalled.
            if os.getpgid(process.pid) != process.pid:
                job["cleanup_errors"].append(label + "_process_group_mismatch")
                return
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except ProcessLookupError:
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                job["cleanup_errors"].append(label + "_stop_not_confirmed")
        except (OSError, subprocess.TimeoutExpired):
            job["cleanup_errors"].append(label + "_stop_not_confirmed")

    def run_job(self, job: dict) -> bool:
        env = self.environment(job)
        info = self.prepared[job["job_id"]]
        folder = Path(job["output_root"])
        token, client, monitor = None, None, None
        handles = []
        job.update(status="starting", started_at=now())
        self.save()
        try:
            folder.mkdir(parents=True, exist_ok=False)
            require(sha_file(info["plan"]) == job["plan_sha256"], "plan_changed_after_validation")
            for ref in info["case"]["references"]:
                require(sha_file(within(ref["remote_path"], self.root)) == ref["sha256"], "source_changed_after_validation")
            require(self.status(env).get("status") == "STOPPED", "fresh_service_requires_stopped")
            deadline = time.monotonic() + READY_TIMEOUT
            rc, state, diagnostic = self.control("start", env, timeout=READY_TIMEOUT)
            job["start_diagnostic_sha256"] = diagnostic
            require(rc == 0 and isinstance(state, dict), "controller_start_failed")
            job["service"] = state
            require(not state.get("reused"), "service_reuse_refused")
            token = state.get("owner_token")
            require(isinstance(token, str) and bool(token), "new_service_owner_missing")
            self.save()
            while time.monotonic() < deadline:
                if self.interrupted:
                    raise InterruptedError("runner_interrupted")
                state = self.status(env, timeout=min(20, max(0.1, deadline - time.monotonic())))
                require(state.get("owner_token") == token, "service_owner_or_process_changed_during_startup")
                if state.get("status") == "READY":
                    break
                require(state.get("status") != "IDENTITY_MISMATCH", "server_identity_mismatch")
                time.sleep(min(STATUS_INTERVAL, max(0, deadline - time.monotonic())))
            else:
                raise TimeoutError("service_ready_deadline")
            job.update(status="running", ready_at=now())
            job["resources"] = str(folder / "resources.jsonl")
            self.save()
            for label in ("monitor", "client"):
                handle = (folder / (label + ".log")).open("x")
                handles.append(handle)
            monitor = subprocess.Popen([sys.executable, str(HERE / "monitor_omni_resources.py"),
                "--project-root", str(self.root), "--interval", "1", "--output", job["resources"]],
                cwd=self.root, env=env, stdin=subprocess.DEVNULL, stdout=handles[0], stderr=subprocess.STDOUT,
                start_new_session=True)
            client = subprocess.Popen([sys.executable, str(HERE / "bench_omni.py"), "--plan", str(info["plan"]),
                "--server", f"http://127.0.0.1:{state['port']}", "--model", state["model_name"],
                "--output-root", str(folder), "--timeout", str(CLIENT_TIMEOUT)],
                cwd=self.root, env=env, stdin=subprocess.DEVNULL, stdout=handles[1], stderr=subprocess.STDOUT,
                start_new_session=True)
            job.update(client_pid=client.pid, monitor_pid=monitor.pid)
            self.save()
            while client.poll() is None:
                if self.interrupted:
                    raise InterruptedError("runner_interrupted")
                require(monitor.poll() is None, "resource_monitor_exited_early")
                time.sleep(1)
            job["client_returncode"] = client.returncode
            reports = sorted(folder.glob("bench_*/report.json"))
            require(len(reports) == 1, "benchmark_report_missing_or_ambiguous")
            job["report"] = str(reports[0])
            report = read_object(reports[0])
            require(report.get("plan_sha256") == job["plan_sha256"], "benchmark_report_plan_mismatch")
            complete = (report.get("planned_requests") == info["request_count"] and
                        len(report.get("plan_ledger", [])) == info["request_count"] and
                        all(row.get("ok") and row.get("outcome") == "succeeded" for row in report.get("plan_ledger", [])) and
                        report.get("completed_requests") == info["request_count"] and
                        report.get("failed_requests") == 0 and report.get("not_run_requests") == 0 and
                        report.get("in_progress_requests") == 0)
            job["status"] = "passed" if client.returncode == 0 and complete else "failed"
            if job["status"] == "failed":
                job["failure_stage"] = "generation"
        except InterruptedError:
            job["status"] = "interrupted"
        except (OSError, ValueError, KeyError, TypeError, TimeoutError, subprocess.SubprocessError) as error:
            job["failure"] = {"type": type(error).__name__, "code": str(error) if isinstance(error, (ValueError, TimeoutError)) else "operation_failed"}
            job["status"] = "failed"
            job["failure_stage"] = "startup" if job["ready_at"] is None else "runner"
        finally:
            self.stop_child(client, "client", job)
            self.stop_child(monitor, "monitor", job)
            if client is not None:
                job["client_returncode"] = client.poll()
            if monitor is not None:
                job["monitor_returncode"] = monitor.poll()
                if monitor.returncode not in (0, None):
                    job["cleanup_errors"].append("resource_monitor_failed")
            for handle in handles:
                handle.close()
            reports = sorted(folder.glob("bench_*/report.json"))
            if len(reports) == 1:
                job["report"] = str(reports[0])
            if token:
                self.stop_service(job, env, token)
            else:
                try:
                    self.final_state(job, self.status(env))
                except (OSError, ValueError, subprocess.SubprocessError):
                    job["cleanup_errors"].append("service_state_unknown")
            if job.get("service_final_state") != "STOPPED":
                job["cleanup_errors"].append("service_not_confirmed_stopped")
            job["finished_at"] = now()
            self.save()
        may_continue = job["status"] == "passed" or (job["status"] == "failed" and job.get("failure_stage") == "generation")
        return may_continue and not job["cleanup_errors"]

    def run(self) -> int:
        self.save()
        for job in self.record["jobs"]:
            if self.interrupted or not self.run_job(job):
                self.record["status"] = "interrupted" if self.interrupted else "aborted"
                break
        else:
            self.record["status"] = ("completed" if all(j["status"] == "passed" for j in self.record["jobs"])
                                     else "completed_with_case_failures")
        self.record["finished_at"] = now()
        self.save()
        return 0 if self.record["status"] == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("/home/isaac/workdir/h3"))
    args = parser.parse_args()
    os.umask(0o077)
    try:
        require(sys.platform == "linux", "linux_required_for_owned_service_lifecycle")
        root = args.project_root.expanduser().resolve()
        protocol_path = args.protocol.expanduser().resolve()
        record_path = args.record.expanduser().resolve()
        output = args.output_root.expanduser().resolve()
        require(not record_path.exists() and not output.exists(), "new_record_and_output_paths_required")
        protocol, prepared = validate_protocol(protocol_path, root)
        runner = Runner(root, record_path, output, protocol_path, protocol, prepared)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=False)
        # Reserve this run record; no automatic resume or overwrite of attempts.
        fd = os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        def interrupt(_signum, _frame):
            runner.interrupted = True
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        code = runner.run()
        print(json.dumps({"status": runner.record["status"], "record": str(record_path)}))
        return code
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "preflight_failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
