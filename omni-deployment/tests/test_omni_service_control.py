"""CPU-only tests; local loopback sockets only, no GPU, Ray or service startup."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import errno
import json
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

    def test_memory_options_default_to_resident_vae_and_native_allocator(self):
        for profile in ("sp1", "sp2"):
            with self.subTest(profile=profile), patch.dict(os.environ, self.asset_env(Path("/deploy/h3")), clear=True):
                config, env, _ = control.settings(profile)
            self.assertIs(config["generator"]["engine"]["offload"]["vae"], False)
            self.assertEqual(env["PYTORCH_ALLOC_CONF"], "backend:native")

    def test_mm_and_stage_defaults_preserve_upstream_baseline(self):
        for profile in ("sp1", "sp2"):
            with self.subTest(profile=profile), patch.dict(os.environ, self.asset_env(Path("/deploy/h3")), clear=True):
                _, env, _ = control.settings(profile)
            self.assertEqual(env["FASTVIDEO_NVFP4_MM_BACKEND"], "auto")
            self.assertEqual(env["FASTVIDEO_NVFP4_MM_ROW_SPLIT"], "off")
            self.assertEqual(env["FASTVIDEO_STAGE_LOGGING"], "1")

    def test_mm_and_stage_options_are_strict(self):
        for name, values in (("OMNI_NVFP4_MM_BACKEND", ("", "AUTO", "triton", " cudnn", "cutlass ")),
                             ("OMNI_STAGE_LOGGING", ("", "true", "false", "2", " 1", "0 "))):
            for value in values:
                with self.subTest(name=name, value=value), patch.dict(os.environ, {
                        **self.asset_env(Path("/deploy/h3")), name: value}, clear=True):
                    with self.assertRaisesRegex(ValueError, name):
                        control.settings("sp2")

    def test_validation_cannot_inherit_mm_stage_or_row_split_overrides(self):
        for backend in ("auto", "cutlass", "cudnn"):
            for stage_logging in ("0", "1"):
                shell_env = {**self.asset_env(Path("/deploy/h3")),
                             "OMNI_NVFP4_MM_BACKEND": backend, "OMNI_STAGE_LOGGING": stage_logging,
                             "FASTVIDEO_NVFP4_MM_BACKEND": "unrequested",
                             "FASTVIDEO_NVFP4_MM_ROW_SPLIT": "auto", "FASTVIDEO_STAGE_LOGGING": "wrong"}
                with self.subTest(backend=backend, stage_logging=stage_logging), \
                        patch.dict(os.environ, shell_env, clear=True), patch.object(control.subprocess, "run") as run:
                    _, env, paths = control.settings("sp2")
                    control.validate(Path("/deploy/h3/run/config.json"), paths, env)
                effective = run.call_args.kwargs["env"]
                self.assertEqual(effective["FASTVIDEO_NVFP4_MM_BACKEND"], backend)
                self.assertEqual(effective["FASTVIDEO_STAGE_LOGGING"], stage_logging)
                self.assertEqual(effective["FASTVIDEO_NVFP4_MM_ROW_SPLIT"], "off")

    def test_actual_start_fingerprints_each_mm_and_stage_combination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run/omni"
            run.mkdir(parents=True)
            fingerprints = set()
            captured = {}

            def popen(argv, **kwargs):
                captured.update(argv=argv, **kwargs)
                return NS(pid=123, poll=lambda: None)

            def proc_identity(pid):
                return {"start_ticks": "42", "cmdline": [arg.encode() for arg in captured["argv"]]}

            for backend in ("auto", "cutlass", "cudnn"):
                for stage_logging in ("0", "1"):
                    with self.subTest(backend=backend, stage_logging=stage_logging), \
                            patch.dict(os.environ, {**self.asset_env(root), "OMNI_NVFP4_MM_BACKEND": backend,
                                "OMNI_STAGE_LOGGING": stage_logging, "FASTVIDEO_NVFP4_MM_ROW_SPLIT": "table"}, clear=True), \
                            patch.object(control, "asset_identity", return_value={"verified": True}), \
                            patch.object(control, "inspect_patch", return_value={"state": "native"}), \
                            patch.object(control, "sha256", return_value="a" * 64), \
                            patch.object(control, "read_state", return_value=None), \
                            patch.object(control.socket, "socket"), patch.object(control, "validate"), \
                            patch.object(control.subprocess, "Popen", side_effect=popen), \
                            patch.object(control, "proc_identity", side_effect=proc_identity):
                        result = control.start("sp2", run)
                    fingerprints.add(result["fingerprint"])
                    effective = captured["env"]
                    self.assertEqual(effective["FASTVIDEO_NVFP4_MM_BACKEND"], backend)
                    self.assertEqual(effective["FASTVIDEO_STAGE_LOGGING"], stage_logging)
                    self.assertEqual(effective["FASTVIDEO_NVFP4_MM_ROW_SPLIT"], "off")
                    identity = json.loads(Path(result["log"]).read_text().splitlines()[0])["identity"]
                    self.assertEqual(identity["env"]["FASTVIDEO_NVFP4_MM_BACKEND"], backend)
                    self.assertEqual(identity["env"]["FASTVIDEO_STAGE_LOGGING"], stage_logging)
            self.assertEqual(len(fingerprints), 6)

    def test_memory_options_change_only_requested_recipe_fields(self):
        for profile in ("sp1", "sp2"):
            with self.subTest(profile=profile), patch.dict(os.environ, self.asset_env(Path("/deploy/h3")), clear=True):
                baseline_config, baseline_env, baseline_paths = control.settings(profile)
                os.environ.update(OMNI_VAE_OFFLOAD="1", OMNI_EXPANDABLE_SEGMENTS="1")
                config, env, paths = control.settings(profile)
            self.assertIs(config["generator"]["engine"]["offload"]["vae"], True)
            self.assertEqual(env["PYTORCH_ALLOC_CONF"], "expandable_segments:True")
            config["generator"]["engine"]["offload"]["vae"] = False
            env["PYTORCH_ALLOC_CONF"] = "backend:native"
            self.assertEqual((config, env, paths), (baseline_config, baseline_env, baseline_paths))

    def test_memory_flags_require_exact_zero_or_one(self):
        for name in ("OMNI_VAE_OFFLOAD", "OMNI_EXPANDABLE_SEGMENTS"):
            for value in ("true", "false", "2", "-1", "", " 1", "1 "):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {
                        **self.asset_env(Path("/deploy/h3")), name: value}, clear=True):
                    with self.assertRaisesRegex(ValueError, name + " must be 0 or 1"):
                        control.settings("sp2")

    def test_reference_policies_preserve_native_and_accept_720_cap(self):
        for policy, cap in (("native", "0"), ("original", "0"), ("cap", "480"), ("cap", "720")):
            with self.subTest(policy=policy, cap=cap), patch.dict(os.environ, {
                    **self.asset_env(Path("/deploy/h3")), "OMNI_REFERENCE_IMAGE_RESIZE_POLICY": policy,
                    "OMNI_REFERENCE_IMAGE_MAX_SHORT_EDGE": cap}, clear=True):
                config, _, _ = control.settings("sp2")
            fields = config["generator"]["pipeline"]["experimental"]
            self.assertEqual(fields.get("h3_reference_image_resize_policy", "native"), policy)
            self.assertEqual(fields.get("h3_reference_image_max_short_edge"), int(cap) if policy == "cap" else None)

    def test_reference_invalid_or_ambiguous_policy_rejected(self):
        for policy, cap in (("cap", "0"), ("cap", "31"), ("cap", "0480"), ("cap", "480.0"),
                            ("cap", " 480"), ("cap", ""), ("native", "480"), ("original", "720"), ("bad", "0")):
            with self.subTest(policy=policy, cap=cap), patch.dict(os.environ, {
                    **self.asset_env(Path("/deploy/h3")), "OMNI_REFERENCE_IMAGE_RESIZE_POLICY": policy,
                    "OMNI_REFERENCE_IMAGE_MAX_SHORT_EDGE": cap}, clear=True), self.assertRaises(ValueError):
                control.settings("sp2")

    def test_validation_subprocess_cannot_inherit_allocator_aliases(self):
        for enabled, expected in (("0", "backend:native"), ("1", "expandable_segments:True")):
            shell_env = {**self.asset_env(Path("/deploy/h3")), "OMNI_EXPANDABLE_SEGMENTS": enabled,
                         "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
                         "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
            with self.subTest(enabled=enabled), patch.dict(os.environ, shell_env, clear=True), \
                    patch.object(control.subprocess, "run") as run:
                _, env, paths = control.settings("sp2")
                control.validate(Path("/deploy/h3/run/config.json"), paths, env)
            child_env = run.call_args.kwargs["env"]
            self.assertEqual(child_env["PYTORCH_ALLOC_CONF"], expected)
            self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", child_env)

    def test_start_sanitizes_allocator_and_fingerprints_both_memory_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run/omni"
            run.mkdir(parents=True)
            fingerprints = set()
            for vae, expandable in (("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")):
                captured = {}

                def popen(argv, **kwargs):
                    captured.update(argv=argv, **kwargs)
                    return NS(pid=123, poll=lambda: None)

                def proc_identity(pid):
                    return {"start_ticks": "42", "cmdline": [arg.encode() for arg in captured["argv"]]}

                shell_env = {**self.asset_env(root), "OMNI_VAE_OFFLOAD": vae,
                             "OMNI_EXPANDABLE_SEGMENTS": expandable,
                             "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
                             "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
                with self.subTest(vae=vae, expandable=expandable), \
                        patch.dict(os.environ, shell_env, clear=True), \
                        patch.object(control, "asset_identity", return_value={"verified": True}), \
                        patch.object(control, "inspect_patch", return_value={"state": "native"}), \
                        patch.object(control, "sha256", return_value="a" * 64), \
                        patch.object(control, "read_state", return_value=None), \
                        patch.object(control.socket, "socket"), patch.object(control, "validate"), \
                        patch.object(control.subprocess, "Popen", side_effect=popen), \
                        patch.object(control, "proc_identity", side_effect=proc_identity):
                    result = control.start("sp2", run)
                fingerprints.add(result["fingerprint"])
                expected = "expandable_segments:True" if expandable == "1" else "backend:native"
                self.assertEqual(captured["env"]["PYTORCH_ALLOC_CONF"], expected)
                self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", captured["env"])
                config = json.loads(Path(result["config"]).read_text())
                self.assertIs(config["generator"]["engine"]["offload"]["vae"], vae == "1")
            self.assertEqual(len(fingerprints), 4)


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


class StopOwnerTokenTest(unittest.TestCase):
    def test_explicit_stop_cleans_exited_owned_record_without_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run/omni"
            run.mkdir(parents=True)
            state = {"pid": 123, "owner_token": "original-token"}
            (run / "service.json").write_text(json.dumps(state))
            with patch.dict(os.environ, {"H3_OMNI_ROOT": directory}, clear=True), \
                    patch.object(control.sys, "platform", "linux"), \
                    patch.object(control.sys, "argv", ["omni_control.py", "stop", "--expected-owner-token", "original-token"]), \
                    patch.object(control, "owned", return_value=False), \
                    patch.object(control.os, "kill") as kill, patch.object(control.os, "killpg") as killpg, \
                    redirect_stdout(io.StringIO()):
                control.main()
            kill.assert_not_called()
            killpg.assert_not_called()
            self.assertFalse((run / "service.json").exists())
            self.assertTrue((run / "stopped-original-token.json").exists())

    def test_main_rejects_missing_or_replaced_state_before_ownership_or_signals(self):
        for state in (None, {"owner_token": "replacement-token"}):
            for force in (False, True):
                with self.subTest(state=state, force=force), tempfile.TemporaryDirectory() as directory:
                    argv = ["omni_control.py", "stop", "--expected-owner-token", "original-token"]
                    if force:
                        argv.append("--force")
                    with patch.dict(os.environ, {"H3_OMNI_ROOT": directory}, clear=True), \
                            patch.object(control.sys, "platform", "linux"), \
                            patch.object(control.sys, "argv", argv), \
                            patch.object(control, "read_state", return_value=state), \
                            patch.object(control, "owned") as owned, \
                            patch.object(control.os, "kill") as kill, \
                            patch.object(control.os, "killpg") as killpg:
                        with self.assertRaisesRegex(RuntimeError, "Expected owner token.*refusing stop"):
                            control.main()
                        owned.assert_not_called()
                        kill.assert_not_called()
                        killpg.assert_not_called()

    def test_main_matching_token_and_manual_stop_keep_existing_stop_behavior(self):
        for expected_token in (None, "original-token"):
            for force in (False, True):
                with self.subTest(expected_token=expected_token, force=force), \
                        tempfile.TemporaryDirectory() as directory:
                    run = Path(directory) / "run/omni"
                    run.mkdir(parents=True)
                    state = {"pid": 123, "pgid": 123, "owner_token": "original-token"}
                    (run / "service.json").write_text(json.dumps(state))
                    argv = ["omni_control.py", "stop"]
                    if expected_token is not None:
                        argv.extend(["--expected-owner-token", expected_token])
                    if force:
                        argv.append("--force")
                    output = io.StringIO()
                    with patch.dict(os.environ, {"H3_OMNI_ROOT": directory}, clear=True), \
                            patch.object(control.sys, "platform", "linux"), \
                            patch.object(control.sys, "argv", argv), \
                            patch.object(control, "owned", side_effect=[True, False, False]), \
                            patch.object(control.os, "kill") as kill, \
                            patch.object(control.os, "killpg") as killpg, redirect_stdout(output):
                        control.main()
                    if force:
                        kill.assert_not_called()
                        killpg.assert_called_once_with(123, control.signal.SIGKILL)
                    else:
                        kill.assert_called_once_with(123, control.signal.SIGTERM)
                        killpg.assert_not_called()
                    self.assertEqual(json.loads(output.getvalue())["status"], "STOPPED")
                    self.assertFalse((run / "service.json").exists())
                    self.assertTrue((run / "stopped-original-token.json").is_file())


class PortProbeTest(unittest.TestCase):
    def test_active_listener_is_rejected_even_with_reuseaddr(self):
        with control.socket.socket() as listener:
            listener.setsockopt(control.socket.SOL_SOCKET, control.socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            with self.assertRaises(OSError) as raised:
                control.check_listener_port_available(listener.getsockname()[1])
            self.assertEqual(raised.exception.errno, errno.EADDRINUSE)

    @unittest.skipUnless(sys.platform == "linux", "Linux server-side TIME_WAIT bind semantics")
    def test_closed_listener_with_server_time_wait_allows_restart(self):
        with control.socket.socket() as listener:
            listener.settimeout(2)
            listener.setsockopt(control.socket.SOL_SOCKET, control.socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen(1)
            with control.socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                accepted, _ = listener.accept()
                with accepted:
                    accepted.settimeout(2)
                    # The server initiates FIN, so its port enters TIME_WAIT.
                    accepted.shutdown(control.socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
                    client.shutdown(control.socket.SHUT_WR)
                    self.assertEqual(accepted.recv(1), b"")
        with control.socket.socket() as bare_probe:
            with self.assertRaises(OSError) as raised:
                bare_probe.bind(("127.0.0.1", port))
            self.assertEqual(raised.exception.errno, errno.EADDRINUSE)
        control.check_listener_port_available(port)


if __name__ == "__main__":
    unittest.main()
