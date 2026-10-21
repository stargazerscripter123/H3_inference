"""Isolated lifecycle tests: no Linux services, GPU, network or private media."""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "omni_business_runner", Path(__file__).resolve().parents[1] / "scripts/run_business_eval.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def protocol_fixture(root):
    source = root / "opaque-reference.bin"
    source.write_bytes(b"opaque bytes for a checksum test, not image pixels")
    cases = [{"case_id": f"case-{i:02d}", "prompt_sha256": sha(b"lifecycle fixture"),
              "references": [{"remote_path": str(source), "sha256": sha(source.read_bytes()),
                              "width": 480, "height": 832}]} for i in range(1, 4)]
    jobs = []
    for case in cases:
        for policy_id, (policy, cap) in runner.POLICIES.items():
            job_id = case["case_id"] + "-" + policy_id
            plan = root / (job_id + ".json")
            requests = [{"case_id": case["case_id"], "prompt": "lifecycle fixture", "width": 480,
                         "height": 832, "frames": 362, "seed": seed, "phase": phase,
                         "cache_condition": "cold" if phase == "warmup" else "warm-full-conditioning",
                         "references": [{"type": "image", "source": str(source)}]}
                        for seed, phase in ((9000, "warmup"), (1234, "measured"))]
            plan.write_text(json.dumps({"requests": requests}))
            jobs.append({"job_id": job_id, "case_id": case["case_id"], "policy_id": policy_id,
                         "profile": "sp2", "resize_policy": policy, "max_short_edge": cap,
                         "plan": str(plan), "plan_sha256": sha(plan.read_bytes())})
    protocol = {"schema_version": 1, "tag": "lifecycle-fixture", "expected": runner.EXPECTED,
                "cases": cases, "policies": [{"policy_id": key, "resize_policy": value[0],
                "max_short_edge": value[1]} for key, value in runner.POLICIES.items()], "jobs": jobs}
    path = root / "protocol.json"
    path.write_text(json.dumps(protocol))
    return path


class Lifecycle:
    """Small process/HTTP-boundary stand-ins, not a second runner implementation."""
    def __init__(self, mode):
        self.mode = mode
        self.clock = 0
        self.calls = []
        self.processes = {}
        self.clients = []
        self.active = False
        self.owner = None
        self.stopping = False
        self.stop_polls = 0

    def control(self, _runner, action, env, *extra, timeout=20):
        self.calls.append((action, extra, env))
        if action == "start":
            self.active, self.stopping, self.stop_polls = True, False, 0
            self.owner = "owner-" + str(len(self.calls))
            return 0, dict(self.state("STARTING"), reused=self.mode == "reuse"), "0" * 64
        if action == "stop":
            assert "--force" not in extra
            assert extra[extra.index("--expected-owner-token") + 1] == self.owner
            self.stopping = True
            if self.mode in {"delayed-stop", "stop-timeout", "owner-change"}:
                return 1, None, "0" * 64
            self.active = False
        if not self.active:
            return 0, {"status": "STOPPED"}, "0" * 64
        if self.stopping:
            self.stop_polls += 1
            if self.mode == "delayed-stop" and self.stop_polls >= 3:
                self.active = False
                return 0, {"status": "STOPPED"}, "0" * 64
            if self.mode == "owner-change":
                return 0, {"status": "READY", "owner_token": "unrelated-owner"}, "0" * 64
        status = "STARTING_OR_UNHEALTHY" if self.mode == "ready-timeout" and not self.stopping else "READY"
        return 0, self.state(status), "0" * 64

    def state(self, status):
        return {"status": status, "owner_token": self.owner, "port": 8108, "model_name": "fixture"}

    def sleep(self, seconds):
        self.clock += seconds

    def popen(self, arguments, **_kwargs):
        lifecycle = self
        class Child:
            def __init__(self):
                self.pid = 10000 + len(lifecycle.processes)
                lifecycle.processes[self.pid] = self
                self.returncode = None
                self.polls = 0
                self.is_client = Path(arguments[1]).name == "bench_omni.py"
                if self.is_client:
                    lifecycle.clients.append(arguments)
                    assert "--server-paths" not in arguments
                    plan = Path(arguments[arguments.index("--plan") + 1])
                    root = Path(arguments[arguments.index("--output-root") + 1])
                    assert root.name.startswith("case-")
                    output = root / "bench_fixture"
                    output.mkdir(parents=True)
                    failed = lifecycle.mode == "generation-failure" and len(lifecycle.clients) == 1
                    self.exitcode = int(failed)
                    ledger = ([{"ok": False, "outcome": "failed"}, {"ok": False, "outcome": "not_run"}]
                              if failed else [{"ok": True, "outcome": "succeeded"}] * 2)
                    report = {"plan_sha256": sha(plan.read_bytes()), "planned_requests": 2,
                              "completed_requests": 1 if failed else 2, "failed_requests": int(failed),
                              "not_run_requests": int(failed), "in_progress_requests": 0, "plan_ledger": ledger}
                    (output / "report.json").write_text(json.dumps(report))

            def poll(self):
                if self.returncode is None and self.is_client:
                    self.polls += 1
                    if lifecycle.mode == "interrupted" and self.polls == 1:
                        lifecycle.runner.interrupted = True
                    if self.polls > 1:
                        self.returncode = self.exitcode
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 0
                return 0
        return Child()

    def killpg(self, pid, sig):
        self.processes[pid].returncode = 0


class BusinessRunnerTests(unittest.TestCase):
    def run_scenario(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            protocol_path = protocol_fixture(root)
            protocol, prepared = runner.validate_protocol(protocol_path, root)
            with patch.object(runner.subprocess, "check_output", return_value="a" * 40), patch.object(
                    runner, "process_identity", return_value={"pid": 1, "pgid": 1, "uid": 1, "start_ticks": "1"}):
                instance = runner.Runner(root, root / "record.json", root / "output", protocol_path, protocol, prepared)
            fake = Lifecycle(mode)
            fake.runner = instance
            with ExitStack() as stack:
                stack.enter_context(patch.object(runner.Runner, "control", lambda instance, *a, **k: fake.control(instance, *a, **k)))
                stack.enter_context(patch.object(runner.subprocess, "Popen", fake.popen))
                stack.enter_context(patch.object(runner.time, "monotonic", lambda: fake.clock))
                stack.enter_context(patch.object(runner.time, "sleep", fake.sleep))
                stack.enter_context(patch.object(runner.os, "getpgid", lambda pid: pid))
                stack.enter_context(patch.object(runner.os, "killpg", fake.killpg))
                code = instance.run()
            record = json.loads((root / "record.json").read_text())
            self.assertIsInstance(record["started_at"], float)
            self.assertEqual(len(record["jobs"]), 9)
            return code, record, fake

    def test_fresh_service_each_job_and_collector_contract(self):
        code, record, fake = self.run_scenario("normal")
        self.assertEqual(code, 0)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(len(fake.clients), 9)
        self.assertEqual(sum(a == "start" for a, _, _ in fake.calls), 9)
        self.assertEqual(sum(a == "stop" for a, _, _ in fake.calls), 9)
        for job in record["jobs"]:
            self.assertEqual(job["service_final_state"], "STOPPED")
            self.assertEqual(Path(job["report"]).parent.parent, Path(job["output_root"]))
            self.assertEqual(job["cleanup_errors"], [])
        for action, _, env in fake.calls:
            if action == "start":
                self.assertEqual((env["OMNI_VAE_OFFLOAD"], env["OMNI_EXPANDABLE_SEGMENTS"], env["OMNI_SP_PROFILE"]), ("1", "1", "0"))

    def test_delayed_stop_polls_without_resending_stop(self):
        code, _, fake = self.run_scenario("delayed-stop")
        self.assertEqual(code, 0)
        self.assertEqual(sum(a == "stop" for a, _, _ in fake.calls), 9)
        self.assertGreaterEqual(fake.clock, 9 * 60)

    def test_generation_failure_is_recorded_never_retried(self):
        code, record, fake = self.run_scenario("generation-failure")
        self.assertEqual(code, 1)
        self.assertEqual(record["status"], "completed_with_case_failures")
        self.assertEqual((record["jobs"][0]["status"], record["jobs"][0]["failure_stage"]), ("failed", "generation"))
        self.assertEqual(len(fake.clients), 9)
        plans = [a[a.index("--plan") + 1] for a in fake.clients]
        self.assertEqual(len(set(plans)), 9)

    def test_readiness_deadline_never_launches_client(self):
        code, record, fake = self.run_scenario("ready-timeout")
        self.assertEqual(code, 1)
        self.assertEqual(len(fake.clients), 0)
        self.assertEqual(record["jobs"][0]["failure_stage"], "startup")
        self.assertEqual(record["jobs"][1]["status"], "not_run")

    def test_unconfirmed_stop_aborts_remaining_jobs(self):
        code, record, fake = self.run_scenario("stop-timeout")
        self.assertEqual(code, 1)
        self.assertEqual(len(fake.clients), 1)
        self.assertIn("service_stop_deadline_not_confirmed", record["jobs"][0]["cleanup_errors"])
        self.assertEqual(record["jobs"][1]["status"], "not_run")

    def test_reused_service_is_never_stopped_by_runner(self):
        code, record, fake = self.run_scenario("reuse")
        self.assertEqual(code, 1)
        self.assertEqual(len(fake.clients), 0)
        self.assertFalse(any(a == "stop" for a, _, _ in fake.calls))
        self.assertEqual(record["jobs"][1]["status"], "not_run")

    def test_owner_change_during_shutdown_refuses_further_signals(self):
        code, record, fake = self.run_scenario("owner-change")
        self.assertEqual(code, 1)
        self.assertEqual(len(fake.clients), 1)
        self.assertEqual(sum(a == "stop" for a, _, _ in fake.calls), 1)
        self.assertIn("service_owner_changed_refused_signals", record["jobs"][0]["cleanup_errors"])

    def test_interruption_preserves_report_and_cleans_up(self):
        code, record, fake = self.run_scenario("interrupted")
        self.assertEqual(code, 1)
        self.assertEqual(record["status"], "interrupted")
        self.assertEqual(len(fake.clients), 1)
        self.assertIsNotNone(record["jobs"][0]["report"])
        self.assertEqual(record["jobs"][0]["service_final_state"], "STOPPED")

    def test_plan_and_source_checksum_mismatch_fail_preflight(self):
        for target in ("plan", "source"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                path = protocol_fixture(root)
                protocol = json.loads(path.read_text())
                damaged = Path(protocol["jobs"][0]["plan"] if target == "plan" else protocol["cases"][0]["references"][0]["remote_path"])
                damaged.write_bytes(damaged.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "sha256_mismatch"):
                    runner.validate_protocol(path, root)

    def test_child_disappearance_wait_timeout_does_not_escape_cleanup(self):
        from unittest.mock import Mock
        child = Mock(pid=123)
        child.poll.return_value = None
        child.wait.side_effect = runner.subprocess.TimeoutExpired("fixture", 5)
        job = {"cleanup_errors": []}
        with patch.object(runner.os, "getpgid", side_effect=ProcessLookupError):
            runner.Runner.stop_child(child, "client", job)
        self.assertEqual(job["cleanup_errors"], ["client_stop_not_confirmed"])


if __name__ == "__main__":
    unittest.main()
