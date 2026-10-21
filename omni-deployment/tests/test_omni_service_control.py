"""CPU-only safety/compatibility tests; no GPU, Ray, network or service startup."""
from pathlib import Path
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import omni_control as control
from omni_ray_health import install_ray_health_adapter
from omni_service import private_ray_node_address


def actor(identity):
    return NS(_actor_id=NS(hex=lambda: identity))


class RayNodeAddressTest(unittest.TestCase):
    def runtime(self, nodes):
        return NS(get_runtime_context=lambda: NS(get_node_id=lambda: "owned"), nodes=lambda: nodes)

    def test_uses_actual_current_node_resource_not_loopback_or_other_node(self):
        nodes = [{"Alive": True, "NodeID": "other", "NodeManagerAddress": "192.0.2.9",
                  "Resources": {"node:192.0.2.9": 1}},
                 {"Alive": True, "NodeID": "owned", "NodeManagerAddress": "192.0.2.10",
                  "Resources": {"node:192.0.2.10": 1}}]
        self.assertEqual(private_ray_node_address(self.runtime(nodes)), "192.0.2.10")

    def test_missing_dead_or_unmatched_resource_fails_before_scheduling(self):
        for nodes in ([], [{"Alive": False, "NodeID": "owned"}],
                      [{"Alive": True, "NodeID": "owned", "NodeManagerAddress": "127.0.0.1",
                        "Resources": {"node:192.0.2.10": 1}}]):
            with self.subTest(nodes=nodes), self.assertRaises(RuntimeError):
                private_ray_node_address(self.runtime(nodes))


class ToolchainSettingsTest(unittest.TestCase):
    def asset_env(self, root):
        return {"H3_OMNI_ROOT": str(root), "OMNI_MODEL_DIR": str(root / "models/qad"),
                "OMNI_ADALN_TABLE": str(root / "tables/adaln.pt"),
                "OMNI_DECODER": str(root / "models/decoder.safetensors"),
                "PATH": "/usr/bin:/bin", "CUDA_HOME": "/old/cuda", "CUDACXX": "/old/cuda/bin/nvcc"}

    def test_default_toolchain_overrides_stale_shell_cuda(self):
        root = Path("/home/isaac/workdir/h3")
        with patch.dict(os.environ, self.asset_env(root), clear=True):
            _, env, paths = control.settings("sp2")
        cuda = root / "envs/omni/lib/python3.12/site-packages/nvidia/cu13"
        self.assertEqual(env["CUDA_HOME"], str(cuda))
        self.assertEqual(env["CUDACXX"], str(cuda / "bin/nvcc"))
        self.assertEqual(env["PATH"].split(os.pathsep)[:3], [str(root / "envs/omni/bin"),
                         str(cuda / "bin"), str(root / ".tools/media/bin")])
        self.assertEqual(paths["python"], str(root / "envs/omni/bin/python"))

    def test_venv_symlink_stays_in_venv_and_project_tools_win(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv_bin = root / "custom-env/bin"
            cuda_bin = root / "custom-env/lib/python3.12/site-packages/nvidia/cu13/bin"
            media_bin, fallback = root / ".tools/media/bin", root / "unrelated/bin"
            for folder in (venv_bin, cuda_bin, media_bin, fallback):
                folder.mkdir(parents=True)
            python = venv_bin / "python"
            python.symlink_to(sys.executable)
            for folder, names in ((cuda_bin, ("nvcc",)), (media_bin, ("ffmpeg", "ffprobe")),
                                  (fallback, ("python", "nvcc", "ffmpeg", "ffprobe"))):
                for name in names:
                    path = folder / name
                    path.write_text("#!/bin/sh\nexit 0\n")
                    path.chmod(0o755)
            shell_env = {**self.asset_env(root), "OMNI_PYTHON": str(python), "PATH": str(fallback)}
            with patch.dict(os.environ, shell_env, clear=True):
                _, env, paths = control.settings("sp1")
            self.assertEqual(paths["python"], str(python))
            self.assertEqual(env["CUDA_HOME"], str(cuda_bin.parent))
            for name, folder in (("python", venv_bin), ("nvcc", cuda_bin),
                                 ("ffmpeg", media_bin), ("ffprobe", media_bin)):
                self.assertEqual(shutil.which(name, path=env["PATH"]), str(folder / name))

    def test_explicit_absolute_toolchain_overrides(self):
        root = Path("/deploy/h3")
        with patch.dict(os.environ, {**self.asset_env(root), "OMNI_CUDA_HOME": "/tools/cuda",
                                    "OMNI_MEDIA_BIN": "/tools/media"}, clear=True):
            _, env, _ = control.settings("sp1")
        self.assertEqual(env["CUDA_HOME"], "/tools/cuda")
        self.assertEqual(env["CUDACXX"], "/tools/cuda/bin/nvcc")
        self.assertEqual(env["PATH"].split(os.pathsep)[:3], ["/deploy/h3/envs/omni/bin",
                         "/tools/cuda/bin", "/tools/media"])

    def test_relative_toolchain_override_is_rejected(self):
        for key in ("OMNI_CUDA_HOME", "OMNI_MEDIA_BIN"):
            with self.subTest(key=key), patch.dict(os.environ, {**self.asset_env(Path("/deploy/h3")),
                                                               key: "relative/toolchain"}, clear=True):
                with self.assertRaisesRegex(ValueError, key + " must be an absolute path"):
                    control.settings("sp1")


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
