"""CPU-only lifecycle tests; no Comfy/Torch imports, GPU, network or signals."""
from pathlib import Path
import json
import os
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import comfy_control as control
import comfy_service as worker


class ControlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.run = self.root / "run/comfy-omni"
        self.run.mkdir(parents=True)
        self.config = control.configuration(self.root)
        self.token = "a" * 32
        self.state = {"pid": 1234, "pgid": 1234, "uid": os.getuid(), "start_ticks": "99",
                      "owner_token": self.token, "service_script": str(SCRIPTS / "comfy_service.py"),
                      "config": str(self.run / "launch.json"), "comfy_commit": control.PIN,
                      "fingerprint": control.fingerprint(self.config), "port": 8188}

    def identity(self, **changes):
        value = {"uid": os.getuid(), "pgid": 1234, "start_ticks": "99",
                 "cmdline": [b"python", b"-u", self.state["service_script"].encode(), b"--config",
                             self.state["config"].encode(), b"--owner-token", self.token.encode()]}
        value.update(changes)
        return value

    def write_state(self, **extra):
        state = {**self.state, **extra}
        control.atomic_json(self.run / "service.json", state)
        return state

    def test_environment_cannot_inherit_other_engine_settings(self):
        inherited = {"HOME": "/kept-home", "PATH": "/other/venv/bin", "PYTHONPATH": "/other/source",
                     "LD_LIBRARY_PATH": "/other/cuda/lib", "CUDA_HOME": "/other/cuda",
                     "CUDA_VISIBLE_DEVICES": "0,1", "FASTVIDEO_FA4": "1", "RAY_ADDRESS": "auto",
                     "NCCL_DEBUG": "TRACE", "PYTORCH_ALLOC_CONF": "bad", "OMNI_PORT": "8108"}
        with patch.dict(os.environ, inherited, clear=True):
            env = control.child_environment(self.config)
        self.assertEqual(env["HOME"], "/kept-home")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["VIRTUAL_ENV"], str(self.root / "envs/comfy-omni"))
        for key in inherited.keys() - {"HOME", "PATH", "CUDA_VISIBLE_DEVICES"}:
            self.assertNotIn(key, env)
        self.assertNotIn("/other", env["PATH"])

    def test_configuration_refuses_outside_and_symlink_escaped_directories(self):
        for key, value in (("input_dir", self.root / "credentials"),
                           ("output_dir", self.root / "outputs/omni"),
                           ("input_dir", self.root.parent / "outside")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.configuration(self.root, **{key: str(value)})
        with tempfile.TemporaryDirectory() as outside:
            (self.root / "inputs").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                control.configuration(self.root)

    def test_venv_directory_cannot_alias_fastvideo_environment(self):
        (self.root / "envs/omni").mkdir(parents=True)
        (self.root / "envs/comfy-omni").symlink_to(self.root / "envs/omni", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "venv_directory"):
            control.configuration(self.root)

    def test_default_command_is_private_single_gpu_without_preview_or_custom_nodes(self):
        launch = {**self.config, "owner_token": self.token,
                  "temp_dir": str(Path(self.config["temp_base"]) / self.token)}
        control.validate_config(launch)
        argv = control.command_arguments(launch)
        for flag in ("--disable-metadata", "--disable-auto-launch", "--disable-all-custom-nodes"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--preview-method") + 1], "none")
        self.assertEqual(argv[argv.index("--listen") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--port") + 1], "8188")
        self.assertNotIn("--whitelist-custom-nodes", argv)
        self.assertNotIn("--enable-manager", argv)

    def test_identity_requires_all_fields_and_exact_wrapper_token_argument(self):
        with patch.object(control, "proc_identity", return_value=self.identity()):
            self.assertTrue(control.owned(self.state))
        for changes in ({"pgid": 999}, {"start_ticks": "100"}, {"uid": os.getuid() + 1},
                        {"cmdline": [b"unrelated", self.token.encode()]}):
            with self.subTest(changes=changes), patch.object(control, "proc_identity", return_value=self.identity(**changes)):
                with self.assertRaisesRegex(RuntimeError, "ownership_mismatch"):
                    control.owned(self.state)
        with patch.object(control, "proc_identity", return_value=None):
            self.assertFalse(control.owned(self.state))

    def test_active_service_is_not_replaced_with_different_recipe(self):
        self.write_state()
        changed = {**self.config, "gpu": 0}
        with patch.object(control, "owned", return_value=True), patch.object(control.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "different_configuration"):
                control.start(changed, self.run)
        popen.assert_not_called()

    def test_busy_port_does_not_launch_or_signal_anything(self):
        with patch.object(control, "check_port", side_effect=OSError("busy")), \
                patch.object(control.subprocess, "Popen") as popen, patch.object(control.os, "kill") as kill:
            with self.assertRaises(OSError):
                control.start(self.config, self.run)
        popen.assert_not_called()
        kill.assert_not_called()

    def test_start_records_ownership_and_creates_fresh_private_log(self):
        captured = {}
        def popen(argv, **kwargs):
            captured.update(argv=argv, **kwargs)
            return SimpleNamespace(pid=1234, poll=lambda: None)
        def identity(pid):
            return {"uid": os.getuid(), "pgid": pid, "start_ticks": "99",
                    "cmdline": [arg.encode() for arg in captured["argv"]]}
        with patch.object(control, "check_port"), patch.object(control, "verify_installation", return_value={}), \
                patch.object(control.subprocess, "Popen", side_effect=popen), \
                patch.object(control, "proc_identity", side_effect=identity):
            result = control.start(self.config, self.run)
        self.assertEqual(result["status"], "STARTING")
        self.assertEqual(result["start_ticks"], "99")
        self.assertEqual(result["gpu"], 1)
        self.assertTrue(captured["start_new_session"])
        self.assertTrue(captured["close_fds"])
        self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(control.read_state(self.run)["owner_token"], result["owner_token"])
        self.assertNotIn(result["owner_token"], Path(result["log"]).read_text())
        self.assertEqual(Path(result["log"]).stat().st_mode & 0o777, 0o600)

    def test_stop_token_mismatch_never_probes_or_signals_process(self):
        self.write_state()
        with patch.object(control, "owned") as owned, patch.object(control.os, "kill") as kill, \
                patch.object(control.os, "killpg") as killpg:
            with self.assertRaisesRegex(RuntimeError, "token_mismatch"):
                control.stop(self.run, "b" * 32, force=True)
        owned.assert_not_called()
        kill.assert_not_called()
        killpg.assert_not_called()

    def test_early_wrapper_exit_quarantines_remaining_group_and_blocks_restart(self):
        process = SimpleNamespace(pid=1234, poll=lambda: 1)
        with patch.object(control, "check_port"), patch.object(control, "verify_installation", return_value={}), \
                patch.object(control.subprocess, "Popen", return_value=process), \
                patch.object(control, "group_members", return_value=[{"pid": 777, "uid": os.getuid()}]), \
                patch.object(control, "proc_identity", return_value=None), \
                patch.object(control.os, "kill") as kill, patch.object(control.os, "killpg") as killpg:
            with self.assertRaisesRegex(RuntimeError, "exited_during_start"):
                control.start(self.config, self.run)
            self.assertEqual(control.status(self.run)["status"], "ORPHANED")
            with self.assertRaisesRegex(RuntimeError, "unverified_start_requires_review"):
                control.start(self.config, self.run)
            state = control.read_state(self.run)
            result = control.stop(self.run, state["owner_token"], force=True, timeout=0)
            self.assertEqual(result["status"], "ORPHANED")
        kill.assert_not_called()
        killpg.assert_not_called()

    def test_stop_timeout_keeps_identity_and_does_not_repeat_term(self):
        self.write_state()
        with patch.object(control, "owned", return_value=True), patch.object(control.os, "kill") as kill:
            first = control.stop(self.run, self.token, timeout=0)
            second = control.stop(self.run, self.token, timeout=0)
        self.assertEqual(first["status"], "STOPPING")
        self.assertEqual(second["status"], "STOPPING")
        kill.assert_called_once_with(1234, signal.SIGTERM)
        self.assertEqual(control.read_state(self.run)["owner_token"], self.token)

    def test_normal_stop_retires_record_only_after_process_and_group_exit(self):
        self.write_state()
        with patch.object(control, "owned", side_effect=[True, True, False, False]), \
                patch.object(control, "group_members", return_value=[]), patch.object(control.os, "kill") as kill:
            result = control.stop(self.run, self.token, timeout=0)
        self.assertEqual(result["status"], "STOPPED")
        kill.assert_called_once_with(1234, signal.SIGTERM)
        self.assertIsNone(control.read_state(self.run))
        self.assertEqual(len(list(self.run.glob("stopped-*.json"))), 1)

    def test_force_requires_group_uid_match(self):
        self.write_state()
        with patch.object(control, "owned", return_value=True), \
                patch.object(control, "group_members", return_value=[{"uid": os.getuid() + 1}]), \
                patch.object(control.os, "killpg") as killpg:
            with self.assertRaisesRegex(RuntimeError, "group_uid_mismatch"):
                control.stop(self.run, self.token, force=True, timeout=0)
        killpg.assert_not_called()

    def test_force_signals_only_verified_owned_group(self):
        self.write_state()
        with patch.object(control, "owned", side_effect=[True, True, False, False]), \
                patch.object(control, "group_members", side_effect=[[{"uid": os.getuid()}], []]), \
                patch.object(control.os, "killpg") as killpg, patch.object(control.os, "kill") as kill:
            result = control.stop(self.run, self.token, force=True, timeout=0)
        killpg.assert_called_once_with(1234, signal.SIGKILL)
        kill.assert_not_called()
        self.assertEqual(result["status"], "STOPPED")

    def test_missing_leader_with_remaining_group_is_orphaned_even_with_force(self):
        self.write_state()
        with patch.object(control, "owned", return_value=False), \
                patch.object(control, "group_members", return_value=[{"uid": os.getuid(), "pid": 888}]), \
                patch.object(control.os, "kill") as kill, patch.object(control.os, "killpg") as killpg:
            result = control.stop(self.run, self.token, force=True, timeout=0)
        self.assertEqual(result["status"], "ORPHANED")
        kill.assert_not_called()
        killpg.assert_not_called()
        self.assertIsNotNone(control.read_state(self.run))

    def test_http_response_without_owned_listener_cannot_be_ready(self):
        with patch.object(control, "listener_owned", return_value=False), \
                patch.object(control.urllib.request, "urlopen") as open_url:
            result = control.probe(self.state)
        self.assertEqual(result["status"], "STARTING_OR_UNHEALTHY")
        open_url.assert_not_called()

    def test_sanitized_output_is_allowlisted(self):
        result = control.sanitized({**self.state, "status": "READY", "prompt": "do not export",
                                    "environment": {"SECRET": "do not export"}, "log": "/private/log"})
        for key in ("owner_token", "config", "service_script", "prompt", "environment", "log"):
            self.assertNotIn(key, result)
        self.assertEqual(result["status"], "READY")

    def test_wrapper_preserves_launch_contract_and_runs_pinned_main_in_process(self):
        launch = {**self.config, "owner_token": self.token,
                  "temp_dir": str(Path(self.config["temp_base"]) / self.token)}
        path = self.run / "launch.json"
        control.atomic_json(path, launch)
        old_argv = sys.argv[:]
        old_path = sys.path[:]
        self.addCleanup(setattr, sys, "argv", old_argv)
        self.addCleanup(setattr, sys, "path", old_path)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), \
                patch.object(worker.os, "chdir"), patch.object(worker.signal, "signal") as handler, \
                patch.object(worker.runpy, "run_path") as run:
            worker.main(["--config", str(path), "--owner-token", self.token])
        run.assert_called_once_with(str(Path(self.config["source"]) / "main.py"), run_name="__main__")
        self.assertEqual(sys.argv, control.command_arguments(launch))
        self.assertEqual(handler.call_args.args[0], signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
