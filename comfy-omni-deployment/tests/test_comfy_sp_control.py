"""SP slot CPU contract tests; no GPU, network, signals, model or media reads."""
from pathlib import Path
import io
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import comfy_control as c
import comfy_service as worker


class SPControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def launch(self, **options):
        config = c.configuration(self.root, instance="sp2", **options)
        return {**config, "owner_token": "a" * 32,
                "temp_dir": str(Path(config["temp_base"]) / ("a" * 32))}

    def vendor_copy(self):
        package = self.root / "isolated-package"
        (package / "scripts").mkdir(parents=True)
        shutil.copytree(SCRIPTS.parent / "custom_nodes", package / "custom_nodes")
        shutil.copyfile(SCRIPTS.parent / "sp-node-contract.json", package / "sp-node-contract.json")
        return package

    def test_sp_visibility_explicit_diagnostics_and_isolation(self):
        config = self.launch(sp_verify=True, sp_profile=True)
        c.validate_config(config)
        self.assertEqual((config["gpu"], config["gpu_ids"], config["port"]), (0, [0, 1], 8191))
        self.assertFalse(config["single_gpu"])
        self.assertEqual(config["parallelism"], "ulysses-sp2")
        inherited = {"MINIMAX_SP_STREAM": "1", "MINIMAX_SP_VERIFY": "bad",
                     "NCCL_P2P_DISABLE": "0", "FASTVIDEO_FA4": "1", "PYTHONPATH": "/other"}
        with patch.dict(os.environ, inherited, clear=True):
            env = c.child_environment(config)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["MINIMAX_SP_DEVICES"], "0,1")
        self.assertEqual(env["MINIMAX_SP_LOGDIR"], config["temp_dir"])
        self.assertEqual((env["MINIMAX_SP_VERIFY"], env["MINIMAX_SP_PROFILE"]), ("1", "1"))
        for key in inherited.keys() - {"MINIMAX_SP_VERIFY"}:
            self.assertNotIn(key, env)
        formal = c.child_environment(self.launch())
        self.assertNotIn("MINIMAX_SP_VERIFY", formal)
        self.assertNotIn("MINIMAX_SP_PROFILE", formal)

    def test_diagnostic_flags_rejected_outside_sp_and_non_boolean_rejected(self):
        for slot in ("default", "gpu0", "gpu1"):
            for name in ("sp_verify", "sp_profile"):
                with self.subTest(slot=slot, name=name), self.assertRaises(ValueError):
                    c.configuration(self.root, instance=slot, **{name: True})
        for value in (1, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.launch(sp_verify=value)
        for key, value in (("gpu_ids", [0]), ("single_gpu", True), ("port", 8188)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "launch_configuration_mismatch"):
                c.validate_config({**self.launch(), key: value})

    def test_only_sp_enables_exact_whitelisted_node(self):
        argv = c.command_arguments(self.launch())
        self.assertEqual(argv.count("--disable-all-custom-nodes"), 1)
        self.assertEqual(argv.count("--whitelist-custom-nodes"), 1)
        self.assertEqual(argv[argv.index("--whitelist-custom-nodes") + 1], "minimax_sp_pdd")
        self.assertEqual(argv[argv.index("--base-directory") + 1], str(SCRIPTS.parent))
        for slot in ("default", "gpu0", "gpu1"):
            config = c.configuration(self.root, instance=slot)
            argv = c.command_arguments({**config, "temp_dir": str(self.root / "temp")})
            self.assertIn("--disable-all-custom-nodes", argv)
            self.assertNotIn("--whitelist-custom-nodes", argv)

    def test_all_sp_modes_use_same_explicit_headroom_and_other_slots_unchanged(self):
        for options in ({}, {"sp_profile": True}, {"sp_verify": True},
                        {"sp_verify": True, "sp_profile": True}):
            with self.subTest(options=options):
                config = self.launch(**options)
                c.validate_config(config)
                self.assertEqual(config["sp_vram_headroom_gib"], 4.0)
                self.assertNotIn("diagnostic_vram_headroom_gib", config)
                argv = c.command_arguments(config)
                self.assertEqual(argv.count("--vram-headroom"), 1)
                self.assertEqual(argv[argv.index("--vram-headroom") + 1], "4.0")
                missing = {key: value for key, value in config.items() if key != "sp_vram_headroom_gib"}
                legacy = {**missing, "diagnostic_vram_headroom_gib": 4.0}
                self.assertNotEqual(c.fingerprint(config), c.fingerprint(missing))
                self.assertNotEqual(c.fingerprint(config), c.fingerprint(legacy))
        for slot in ("default", "gpu0", "gpu1"):
            config = c.configuration(self.root, instance=slot)
            self.assertNotIn("sp_vram_headroom_gib", config)
            self.assertNotIn("diagnostic_vram_headroom_gib", config)
            self.assertNotIn("--vram-headroom", c.command_arguments({**config, "temp_dir": str(self.root / "temp")}))

    def test_headroom_tampering_legacy_and_nonsp_injection_fail_validation(self):
        for options in ({}, {"sp_profile": True}, {"sp_verify": True}):
            config = self.launch(**options)
            missing = {key: value for key, value in config.items() if key != "sp_vram_headroom_gib"}
            invalid = [missing, {**missing, "diagnostic_vram_headroom_gib": 4.0},
                       {**config, "diagnostic_vram_headroom_gib": 4.0}]
            invalid += [{**config, "sp_vram_headroom_gib": value}
                        for value in (None, 0, 1, 2.0, 3, "4.0", float("nan"))]
            for candidate in invalid:
                with self.subTest(options=options, candidate=candidate), self.assertRaisesRegex(ValueError, "launch_configuration_mismatch"):
                    c.validate_config(candidate)
        for slot in ("default", "gpu0", "gpu1"):
            for key in ("sp_vram_headroom_gib", "diagnostic_vram_headroom_gib"):
                for value in (None, 0, 4.0):
                    with self.subTest(slot=slot, key=key, value=value), self.assertRaisesRegex(ValueError, "launch_configuration_mismatch"):
                        c.validate_config({**c.configuration(self.root, instance=slot), key: value})

    def test_headroom_survives_launch_state_and_sanitized_receipt(self):
        config = c.configuration(self.root, instance="sp2")
        captured = {}
        def popen(argv, **kwargs):
            captured.update(argv=argv, **kwargs)
            return SimpleNamespace(pid=1234, poll=lambda: None)
        def identity(pid):
            return {"uid": os.getuid(), "pgid": pid, "start_ticks": "99",
                    "cmdline": [arg.encode() for arg in captured["argv"]]}
        run = Path(config["run_dir"])
        with patch.object(c, "require_available_slot"), patch.object(c, "check_port"), \
                patch.object(c, "verify_installation", return_value={}), \
                patch.object(c.subprocess, "Popen", side_effect=popen), \
                patch.object(c, "proc_identity", side_effect=identity):
            result = c.start(config, run)
        launch = json.loads(Path(result["config"]).read_text())
        for receipt in (launch, c.read_state(run), c.sanitized(result)):
            self.assertEqual(receipt["sp_vram_headroom_gib"], 4.0)
            self.assertNotIn("diagnostic_vram_headroom_gib", receipt)
        self.assertEqual(result["fingerprint"], c.fingerprint(config))
        with patch.object(c, "owned", return_value=True), patch.object(c, "probe", return_value={"status": "READY"}):
            ready = c.sanitized(c.status(run))
        self.assertEqual(ready["status"], "READY")
        self.assertEqual(ready["sp_vram_headroom_gib"], 4.0)

    def test_wrapper_delivers_native_headroom_argv_before_comfy_import(self):
        config = self.launch()
        path = self.root / "launch.json"
        path.write_text(json.dumps(config))
        self.addCleanup(setattr, sys, "argv", sys.argv[:])
        self.addCleanup(setattr, sys, "path", sys.path[:])
        observed = []
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=True), \
                patch.object(c, "verify_sp_node"), patch.object(worker.os, "chdir"), \
                patch.object(worker.signal, "signal"), \
                patch.object(worker.runpy, "run_path", side_effect=lambda *a, **k: observed.append(sys.argv[:])):
            worker.main(["--config", str(path), "--owner-token", config["owner_token"]])
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][observed[0].index("--vram-headroom") + 1], "4.0")

    def test_vendor_inventory_and_changed_source_are_rejected(self):
        package = self.vendor_copy()
        with patch.object(c, "HERE", package / "scripts"):
            evidence = c.verify_sp_node()
            self.assertEqual(evidence["contract_sha256"], c.sha256(package / "sp-node-contract.json"))
            base = package / "custom_nodes/minimax_sp_pdd"
            extra = base / "unlisted.py"
            extra.write_text("# unauthorized extra source\n")
            with self.assertRaisesRegex(RuntimeError, "python_inventory_mismatch"):
                c.verify_sp_node()
            extra.unlink()
            target = base / "minimax_sp/sp_forward.py"
            target.write_text(target.read_text() + "\n# changed\n")
            with self.assertRaisesRegex(RuntimeError, "source_hash_mismatch"):
                c.verify_sp_node()

    def test_vendor_alias_rejected_even_with_same_bytes(self):
        package = self.vendor_copy()
        base = package / "custom_nodes/minimax_sp_pdd"
        target = base / "minimax_sp/sp_worker.py"
        external = self.root / "same-public-source.py"
        external.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(external)
        with patch.object(c, "HERE", package / "scripts"), self.assertRaisesRegex(RuntimeError, "source_hash_mismatch"):
            c.verify_sp_node()

    def test_live_sp_reserves_both_gpus_and_either_gpu_blocks_sp(self):
        for target_slot, occupied_slot in (("gpu0", "sp2"), ("gpu1", "sp2"),
                                           ("default", "sp2"), ("sp2", "gpu0"), ("sp2", "gpu1")):
            target = c.configuration(self.root, instance=target_slot)
            occupied = c.configuration(self.root, instance=occupied_slot)
            state = {**occupied, "identity_established": True, "pgid": 1234}
            def lookup(run):
                return state if run == Path(occupied["run_dir"]) else None
            with self.subTest(target=target_slot, occupied=occupied_slot), \
                    patch.object(c, "read_state", side_effect=lookup), patch.object(c, "owned", return_value=True), \
                    patch.object(c.os, "kill") as kill, patch.object(c.os, "killpg") as killpg:
                with self.assertRaisesRegex(RuntimeError, "gpu_reserved_by_other_instance"):
                    c.require_available_slot(target, Path(target["run_dir"]))
                kill.assert_not_called()
                killpg.assert_not_called()

    def test_readiness_requires_two_visible_gpus_and_sp_node(self):
        state = self.launch()
        for device_ids, sp_present, expected in (([0, 1], True, "READY"), ([0], True, "IDENTITY_MISMATCH"),
                                                   ([1, 0], True, "IDENTITY_MISMATCH"), ([0, 1], False, "NATIVE_NODE_MISSING")):
            urls = []
            def opening(url, timeout):
                urls.append(url)
                if url.endswith("system_stats"):
                    result = {"system": {"comfyui_version": c.VERSION},
                              "devices": [{"type": "cuda", "index": n} for n in device_ids]}
                elif url.endswith("MiniMaxH3SPUNETLoader"):
                    result = {"MiniMaxH3SPUNETLoader": {}} if sp_present else {}
                else:
                    result = {"MiniMaxH3ReferenceToVideo": {}}
                return io.StringIO(json.dumps(result))
            with self.subTest(devices=device_ids, sp=sp_present), patch.object(c, "listener_owned", return_value=True), \
                    patch.object(c.urllib.request, "build_opener", return_value=SimpleNamespace(open=opening)):
                self.assertEqual(c.probe(state)["status"], expected)
                self.assertTrue(all(url.startswith("http://127.0.0.1:8191/") for url in urls))

    def test_wrapper_rejects_single_gpu_visibility_before_loading_code(self):
        config = self.launch()
        path = self.root / "launch.json"
        path.write_text(json.dumps(config))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True), patch.object(worker.runpy, "run_path") as execute:
            with self.assertRaisesRegex(RuntimeError, "gpu_visibility_mismatch"):
                worker.main(["--config", str(path), "--owner-token", config["owner_token"]])
            execute.assert_not_called()

    def test_wrapper_rechecks_vendor_before_loading_core(self):
        config = self.launch()
        path = self.root / "launch.json"
        path.write_text(json.dumps(config))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=True), \
                patch.object(c, "verify_sp_node", side_effect=RuntimeError("source_hash_mismatch")), \
                patch.object(worker.runpy, "run_path") as execute:
            with self.assertRaisesRegex(RuntimeError, "source_hash_mismatch"):
                worker.main(["--config", str(path), "--owner-token", config["owner_token"]])
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
