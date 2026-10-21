"""Metadata-only variant tests: no private protocol, media, GPU or network."""
from contextlib import redirect_stderr
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_optimization_eval as optimization
from test_omni_business_runner import protocol_fixture, sha


def four_request_fixture(root):
    path = protocol_fixture(root)
    protocol = json.loads(path.read_text())
    for job in protocol["jobs"]:
        plan_path = Path(job["plan"])
        plan = json.loads(plan_path.read_text())
        for seed in (1235, 1236):
            plan["requests"].append({**copy.deepcopy(plan["requests"][-1]), "seed": seed})
        plan_path.write_text(json.dumps(plan))
        job["plan_sha256"] = sha(plan_path.read_bytes())
    path.write_text(json.dumps(protocol))
    return path


class OptimizationRunnerTest(unittest.TestCase):
    def test_validates_all_nine_then_selects_unchanged_cap480_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = four_request_fixture(root)
            before = path.read_bytes()
            protocol, prepared = optimization.business.validate_protocol(path, root)
            selected, selected_prepared = optimization.select_jobs(protocol, prepared, None)
            self.assertEqual(len(protocol["jobs"]), 9)
            self.assertEqual(len(selected["jobs"]), 3)
            self.assertEqual({job["policy_id"] for job in selected["jobs"]}, {"cap480"})
            for job in selected["jobs"]:
                self.assertEqual(sha(Path(job["plan"]).read_bytes()), job["plan_sha256"])
                self.assertEqual(selected_prepared[job["job_id"]]["request_count"], 4)
            one, _ = optimization.select_jobs(protocol, prepared, "case-01")
            self.assertEqual([job["case_id"] for job in one["jobs"]], ["case-01"])
            self.assertEqual(path.read_bytes(), before)
            # A bad nonselected policy is still refused by full preflight.
            Path(protocol["jobs"][1]["plan"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "plan_sha256_mismatch"):
                optimization.business.validate_protocol(path, root)

    def test_requires_exact_four_requests_and_distinct_seeds(self):
        for fault in ("count", "seed"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                path = four_request_fixture(root)
                protocol, prepared = optimization.business.validate_protocol(path, root)
                job = next(job for job in protocol["jobs"] if job["policy_id"] == "cap480")
                plan_path = prepared[job["job_id"]]["plan"]
                plan = json.loads(plan_path.read_text())
                if fault == "count":
                    plan["requests"].pop()
                else:
                    plan["requests"][-1]["seed"] = plan["requests"][0]["seed"]
                plan_path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "optimization_requires"):
                    optimization.select_jobs(protocol, prepared, None)

    def test_variants_override_stale_shell_and_keep_original_record_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = four_request_fixture(root)
            protocol, prepared = optimization.business.validate_protocol(path, root)
            selected, prepared = optimization.select_jobs(protocol, prepared, None)
            baseline, decoder8 = root / "baseline.bin", root / "decoder8.bin"
            baseline.write_bytes(b"baseline asset metadata fixture")
            decoder8.write_bytes(b"decoder8 asset metadata fixture")
            shell = {"OMNI_DECODER": str(baseline), "OMNI_DECODER8": str(decoder8),
                     "OMNI_NVFP4_MM_BACKEND": "invalid", "OMNI_STAGE_LOGGING": "bad",
                     "OMNI_VAE_OFFLOAD": "bad", "OMNI_SP_PROFILE": "bad", "OMNI_MEMO_ENTRIES": "0"}
            for variant, recipe in optimization.VARIANTS.items():
                for profile in (0, 1):
                    with self.subTest(variant=variant, profile=profile), patch.dict(os.environ, shell, clear=True):
                        overrides, metadata = optimization.variant_settings(variant, profile, 0)
                        with patch.object(optimization.business.subprocess, "check_output", return_value="a" * 40), \
                                patch.object(optimization.business, "process_identity", return_value={"pid": 1}):
                            instance = optimization.OptimizationRunner(root, root / "record.json", root / "new-output",
                                path, selected, prepared, overrides=overrides, metadata=metadata)
                        env = instance.environment(instance.record["jobs"][0])
                    self.assertEqual(env["OMNI_NVFP4_MM_BACKEND"], recipe["backend"])
                    self.assertEqual(env["OMNI_VAE_OFFLOAD"], "1" if recipe["vae_offload"] else "0")
                    self.assertEqual(env["OMNI_STAGE_LOGGING"], "0")
                    self.assertEqual(env["OMNI_SP_PROFILE"], str(profile))
                    self.assertEqual(env["OMNI_MEMO_ENTRIES"], "16")
                    self.assertEqual(env["OMNI_DECODER"], str(decoder8 if variant == "decoder8" else baseline))
                    self.assertIs(metadata["formal_timing_eligible"], not bool(profile))
                    self.assertEqual(metadata["nvfp4_mm_row_split"], "off")
                    self.assertEqual(instance.record["protocol_sha256"], sha(path.read_bytes()))
                    self.assertEqual(len(instance.record["jobs"]), 3)
                    self.assertTrue(all(job["optimization_variant"] == variant for job in instance.record["jobs"]))
                    self.assertTrue(all(Path(job["output_root"]).is_relative_to(root / "new-output")
                                        for job in instance.record["jobs"]))

    def test_missing_decoder8_and_invalid_runtime_options_fail_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "decoder_asset_required"):
                optimization.variant_settings("decoder8", 0, 1)
        for variant, profile, stage in (("typo", 0, 1), ("baseline", 2, 1),
                                        ("baseline", True, 1), ("baseline", 0, "0")):
            with self.subTest(variant=variant, profile=profile, stage=stage), self.assertRaises(ValueError):
                optimization.variant_settings(variant, profile, stage)

    def test_cli_refuses_existing_or_external_destinations_before_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "outputs/old").mkdir(parents=True)
            (root / "run/omni").mkdir(parents=True)
            old = root / "run/omni/old.json"
            old.write_text("preserve")
            for record, output in ((old, root / "outputs/new"), (root / "run/omni/new.json", root / "outputs/old"),
                                   (root / "elsewhere/new.json", root / "outputs/new"),
                                   (root / "run/omni/new.json", root / "not-outputs/new")):
                args = ["run_optimization_eval.py", "--protocol", str(root / "unused.json"), "--variant", "baseline",
                        "--record", str(record), "--output-root", str(output), "--project-root", str(root)]
                with self.subTest(record=record, output=output), patch.object(sys, "argv", args), \
                        patch.object(sys, "platform", "linux"), redirect_stderr(io.StringIO()), \
                        patch.object(optimization.business, "validate_protocol") as validate, \
                        patch.object(optimization.OptimizationRunner, "run") as run:
                    self.assertEqual(optimization.main(), 2)
                    validate.assert_not_called()
                    run.assert_not_called()
            self.assertEqual(old.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
