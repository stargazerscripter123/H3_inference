"""CPU-only safety/compatibility tests; no GPU, Ray, network or service startup."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import omni_control as control
from omni_ray_health import install_ray_health_adapter


def actor(identity):
    return NS(_actor_id=NS(hex=lambda: identity))


class RayHealthTest(unittest.TestCase):
    def engine(self, states, backend="ray", workers=None):
        class Engine:
            @property
            def healthy(self):
                return not self._closed and self._unhealthy_reason is None and all(
                    worker.proc.is_alive() for worker in self._generator.executor.workers)

            @property
            def unhealthy_reason(self):
                return self._unhealthy_reason if self._unhealthy_reason else (None if self.healthy else "worker failed")

        def lookup(identity):
            value = states[identity]
            if isinstance(value, Exception):
                raise value
            return {"State": value} if value is not None else {}

        install_ray_health_adapter(Engine, lookup)
        obj = Engine()
        obj._closed, obj._unhealthy_reason = False, None
        obj._generator = NS(executor=NS(
            fastvideo_args=NS(distributed_executor_backend=backend, num_gpus=2),
            workers=workers if workers is not None else [actor("a"), actor("b")]))
        return obj

    def test_ray_handles_without_proc_are_healthy(self):
        self.assertTrue(self.engine({"a": "ALIVE", "b": "ALIVE"}).healthy)

    def test_dead_restarting_missing_and_gcs_error_fail_closed(self):
        for bad in ("DEAD", "RESTARTING", None, RuntimeError("GCS down")):
            with self.subTest(bad=bad):
                item = self.engine({"a": "ALIVE", "b": bad})
                self.assertFalse(item.healthy)
                self.assertIsNotNone(item.unhealthy_reason)

    def test_shutdown_existing_error_and_missing_rank_fail(self):
        item = self.engine({"a": "ALIVE", "b": "ALIVE"})
        item._closed = True
        self.assertFalse(item.healthy)
        item._closed, item._unhealthy_reason = False, "broken pipe"
        self.assertFalse(item.healthy)
        self.assertFalse(self.engine({"a": "ALIVE"}, workers=[actor("a")]).healthy)

    def test_mp_still_uses_original_liveness(self):
        workers = [NS(proc=NS(is_alive=lambda: False))]
        self.assertFalse(self.engine({}, backend="mp", workers=workers).healthy)


class OwnershipTest(unittest.TestCase):
    def setUp(self):
        self.state = {"pid": 123, "uid": 501, "start_ticks": "123456",
                      "service_script": "/opt/omni_service.py", "owner_token": "owned-token"}
        self.info = {"uid": 501, "pgid": 123, "start_ticks": "123456",
                     "cmdline": [b"python", b"/opt/omni_service.py", b"--owner-token", b"owned-token"]}

    def test_exact_owned_process_matches(self):
        with patch.object(control, "proc_identity", return_value=self.info), patch.object(control.os, "getuid", return_value=501):
            self.assertTrue(control.owned(self.state))

    def test_reused_pid_wrong_uid_group_script_or_token_refused(self):
        variants = [{"start_ticks": "999"}, {"uid": 502}, {"pgid": 99},
                    {"cmdline": [b"python", b"/opt/unrelated.py", b"owned-token"]},
                    {"cmdline": [b"python", b"/opt/omni_service.py", b"someone-else"]}]
        for variant in variants:
            info = {**self.info, **variant}
            with self.subTest(variant=variant), patch.object(control, "proc_identity", return_value=info), patch.object(control.os, "getuid", return_value=501):
                with self.assertRaisesRegex(RuntimeError, "ownership mismatch"):
                    control.owned(self.state)

    def test_dead_process_not_reused(self):
        with patch.object(control, "proc_identity", return_value=None):
            self.assertFalse(control.owned(self.state))

    def test_virtualenv_interpreter_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "venv-python"
            alias.symlink_to(sys.executable)
            self.assertEqual(control.absolute(str(alias), "OMNI_PYTHON", follow_symlinks=False), alias)


if __name__ == "__main__":
    unittest.main()
