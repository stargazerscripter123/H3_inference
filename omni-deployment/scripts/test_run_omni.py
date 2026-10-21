#!/usr/bin/env python3
"""CPU-only client checks. HTTP is mocked; media is real ffmpeg/ffprobe output.

These establish the client contract and media rejection behavior, not GPU
correctness, reference conditioning, latency, or deployment readiness.
"""
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import bench_omni
import run_omni as client


class RequestTests(unittest.TestCase):
    def test_default_port_matches_launcher(self):
        args = client.parser().parse_args(["--prompt", "test"])
        self.assertEqual(args.server, "http://127.0.0.1:8108")

    def request(self, references, **kwargs):
        values = dict(seed=1234, frames=124, width=832, height=480,
                      model="fasth3-omni", server_paths=True)
        values.update(kwargs)
        return client.build_request("Follow <Picture 1>.", references, **values)

    def test_exact_pdd_payload_and_reference_order(self):
        payload, audit = self.request([("image", "/in/a.png"), ("image", "/in/b.png"),
                                       ("video", "/in/m.mp4"), ("audio", "/in/a.wav")])
        self.assertEqual(payload, {
            "task": "ref2va", "prompt": "Follow <Picture 1>.", "seed": 1234,
            "width": 832, "height": 480, "num_frames": 124, "fps": 24,
            "num_inference_steps": 8, "guidance_scale": 1.0,
            "num_outputs_per_prompt": 1, "model": "fasth3-omni",
            "image_reference": [{"image_url": "/in/a.png"}, {"image_url": "/in/b.png"}],
            "video_reference": [{"video_url": "/in/m.mp4"}],
            "audio_reference": [{"audio_url": "/in/a.wav"}],
        })
        self.assertEqual([x["index_within_type"] for x in audit], [1, 2, 1, 1])
        self.assertTrue(all(x["sha256"] is None for x in audit))

    def test_rejects_interleaving_instead_of_silent_reordering(self):
        with self.assertRaisesRegex(client.ClientError, "interleaving"):
            self.request([("video", "/in/m.mp4"), ("image", "/in/a.png")])

    def test_rejects_audio_only_and_invalid_grid_and_reference_limit(self):
        for references, extras in [([("audio", "/in/a.wav")], {}),
                                    ([("image", "/in/a.png")], {"frames": 360}),
                                    ([("image", "/in/a.png")] * 10, {})]:
            with self.subTest(references=references, extras=extras):
                with self.assertRaises(client.ClientError):
                    self.request(references, **extras)

    def test_local_references_are_hashed_but_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "image.png"
            source.write_bytes(b"source bytes preserved")
            payload, audit = self.request([("image", str(source))], server_paths=False)
            self.assertEqual(payload["image_reference"][0]["image_url"], str(source.resolve()))
            self.assertEqual(audit[0]["sha256"], client.sha256_file(source))
            self.assertEqual(source.read_bytes(), b"source bytes preserved")

    def test_integer_fields_reject_json_booleans_and_floats(self):
        for key, value in (("seed", True), ("seed", 1.5), ("frames", 124.0),
                           ("width", 832.0), ("height", False)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(client.ClientError, "integer"):
                self.request([("image", "/in/a.png")], **{key: value})

    def test_timeout_requires_finite_positive_values(self):
        for value in (float("nan"), float("inf"), 0, -1, True):
            with self.subTest(value=value), self.assertRaisesRegex(client.ClientError, "finite"):
                client.require_positive_finite(timeout=value)

    def test_summary_keeps_failures_in_sla_denominator(self):
        base = {"phase": "measured", "cache_condition": "warm-new", "width": 832, "height": 480, "frames": 362}
        summary = bench_omni.summarize([
            {**base, "ok": True, "client_to_media_s": 10.0},
            {**base, "ok": True, "client_to_media_s": 20.0},
            {**base, "ok": False},
            {**base, "phase": "warmup", "ok": True, "client_to_media_s": 100.0},
        ], 15.0)[0]
        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["valid_on_time_rate"], 1 / 3)
        self.assertEqual(summary["p50_s_successes_only"], 15.0)

    def test_summary_keeps_reference_scenarios_separate(self):
        base = {"phase": "measured", "cache_condition": "warm-new", "width": 832,
                "height": 480, "frames": 362, "ok": True}
        summary = bench_omni.summarize([
            {**base, "case_id": "single-image", "client_to_media_s": 10.0},
            {**base, "case_id": "video-ref", "client_to_media_s": 20.0},
            {**base, "client_to_media_s": 30.0},
        ], 15.0)
        by_case = {row["case_id"]: row for row in summary}
        self.assertEqual(set(by_case), {"single-image", "video-ref", "all-unspecified"})
        self.assertEqual(by_case["single-image"]["p50_s_successes_only"], 10.0)
        self.assertEqual(by_case["video-ref"]["valid_on_time_rate"], 0.0)

    def test_warmup_failure_preserves_full_not_run_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index, (case_id, phase) in enumerate([
                    ("S4", "warmup"), ("S4", "measured"), ("S4", "measured"),
                    ("other-case", "measured"), ("other-case", "measured")]):
                rows.append({"name": f"row-{index}", "case_id": case_id, "phase": phase,
                             "cache_condition": "unspecified" if phase == "warmup" else "warm-new",
                             "prompt": "A test scene.", "seed": index, "frames": 124,
                             "references": [{"type": "image", "source": "/server/input.png"}]})
            plan = root / "plan.json"
            plan.write_text(json.dumps({"requests": rows}))

            def fail(_server, _payload, _audit, **kwargs):
                # The complete ledger is already durable before the first job.
                before = json.loads((kwargs["output_root"] / "report.json").read_text())
                self.assertEqual(len(before["plan_ledger"]), 5)
                self.assertEqual(before["plan_ledger"][0]["outcome"], "in_progress")
                raise client.ClientError("server OOM", manifest={
                    "manifest_path": "/evidence/manifest.json", "request_id": "oom-job",
                    "failed_phase": "failed", "server_may_still_be_running": False,
                    "server_job": {"status": "failed"}})

            argv = ["bench_omni.py", "--plan", str(plan), "--server-paths", "--output-root", str(root / "out")]
            with patch("sys.argv", argv), patch("bench_omni.shutil.which", return_value="ffprobe"), \
                    patch("bench_omni.run_omni.run_job", side_effect=fail) as runner, patch("builtins.print"):
                self.assertEqual(bench_omni.main(), 1)
            self.assertEqual(runner.call_count, 1)
            report = json.loads(next((root / "out").glob("*/report.json")).read_text())
            self.assertEqual(report["planned_requests"], 5)
            self.assertEqual(report["attempted_requests"], 1)
            self.assertEqual(report["failed_requests"], 1)
            self.assertEqual(report["not_run_requests"], 4)
            self.assertEqual(report["attempted_measured_requests"], 0)
            self.assertEqual(report["results"][0]["request_id"], "oom-job")
            self.assertEqual(report["results"][0]["manifest_path"], "/evidence/manifest.json")
            for entry in report["plan_ledger"][1:]:
                self.assertEqual(entry["outcome"], "not_run")
                self.assertEqual(entry["blocked_by_index"], 0)
            self.assertEqual(len(report["summary"]), 2)
            for group in report["summary"]:
                self.assertEqual((group["planned"], group["attempted"], group["failed"], group["not_run"]), (2, 0, 0, 2))
                self.assertIsNone(group["valid_on_time_rate"])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media_dir = tempfile.TemporaryDirectory()
        cls.media = Path(cls.media_dir.name) / "fixture.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=24",
            "-f", "lavfi", "-i", "anullsrc=r=32000:cl=stereo",
            "-t", str(124 / 24), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "32000", "-ac", "2", str(cls.media),
        ], check=True, capture_output=True, timeout=60)
        cls.media_bytes = cls.media.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.media_dir.cleanup()

    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.payload, self.audit = client.build_request(
            "A blue scene.", [("image", "/server/ref.png")], seed=42, frames=124,
            width=32, height=32, model=None, server_paths=True)

    def manifest(self):
        return json.loads(next(Path(self.root.name).glob("*/manifest.json")).read_text())

    def test_submit_poll_download_and_validate(self):
        calls = []
        responses = [
            {"id": "job-123", "status": "queued"},
            {"id": "job-123", "status": "in_progress"},
            {"id": "job-123", "status": "completed", "model": "omni-test",
             "inference_time_s": 0.25, "stage_durations": {"denoise": 0.1}},
            self.media_bytes,
        ]

        def fake_open(request, timeout):
            if hasattr(request, "full_url"):
                calls.append((request.get_method(), request.full_url))
                if request.data:
                    self.assertEqual(json.loads(request.data), self.payload)
            else:
                calls.append(("GET", request))
            value = responses.pop(0)
            return io.BytesIO(value if isinstance(value, bytes) else json.dumps(value).encode())

        with patch("urllib.request.urlopen", side_effect=fake_open):
            result = client.run_job("http://127.0.0.1:8092/v1", self.payload, self.audit,
                                    output_root=Path(self.root.name), poll_interval=0.001)
        self.assertEqual(calls, [("POST", "http://127.0.0.1:8092/v1/videos"),
                                  ("GET", "http://127.0.0.1:8092/v1/videos/job-123"),
                                  ("GET", "http://127.0.0.1:8092/v1/videos/job-123"),
                                  ("GET", "http://127.0.0.1:8092/v1/videos/job-123/content")])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["media_validation"]["frames"], 124)
        self.assertIsNone(result["timings"]["server_full_request_s"])
        self.assertEqual(result["timings"]["server_reported_inference_s"], 0.25)
        self.assertGreaterEqual(result["timings"]["client_to_validated_s"], result["timings"]["client_to_media_s"])
        self.assertEqual(self.manifest()["status"], "completed")

    def test_failed_generation_retains_job_without_resubmission(self):
        terminal = {"id": "failed-job", "status": "failed", "error": {"message": "OOM"}}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(terminal).encode())) as transport:
            with self.assertRaisesRegex(client.ClientError, "OOM"):
                client.run_job("http://127.0.0.1:8092", self.payload, self.audit, output_root=Path(self.root.name))
        self.assertEqual(transport.call_count, 1)
        result = self.manifest()
        self.assertEqual(result["request_id"], "failed-job")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["server_may_still_be_running"])

    def test_ambiguous_submission_is_not_retried(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("network timeout")) as transport:
            with self.assertRaises(client.ClientError):
                client.run_job("http://127.0.0.1:8092", self.payload, self.audit, output_root=Path(self.root.name))
        self.assertEqual(transport.call_count, 1)
        result = self.manifest()
        self.assertEqual(result["status"], "submission_unknown")
        self.assertTrue(result["server_may_still_be_running"])

    def test_rejected_submission_is_not_unknown(self):
        error = urllib.error.HTTPError("http://server/v1/videos", 400, "Bad Request", {}, io.BytesIO(b'{"error":"invalid media"}'))
        with patch("urllib.request.urlopen", side_effect=error) as transport:
            with self.assertRaises(client.ClientError):
                client.run_job("http://127.0.0.1:8092", self.payload, self.audit, output_root=Path(self.root.name))
        self.assertEqual(transport.call_count, 1)
        result = self.manifest()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["server_may_still_be_running"])

    def test_poll_4xx_does_not_prove_running_job_was_rejected(self):
        accepted = io.BytesIO(json.dumps({"id": "active-job", "status": "in_progress"}).encode())
        missing = urllib.error.HTTPError("http://server/v1/videos/active-job", 404, "Not Found", {}, io.BytesIO(b"missing job"))
        with patch("urllib.request.urlopen", side_effect=[accepted, missing]) as transport:
            with self.assertRaises(client.ClientError) as raised:
                client.run_job("http://127.0.0.1:8108", self.payload, self.audit,
                               output_root=Path(self.root.name), poll_interval=0.001)
        self.assertEqual(transport.call_count, 2)
        result = self.manifest()
        self.assertEqual(result["request_id"], "active-job")
        self.assertEqual(result["failed_phase"], "in_progress")
        self.assertTrue(result["server_may_still_be_running"])
        self.assertEqual(raised.exception.manifest["manifest_path"], result["manifest_path"])

    def test_bad_media_is_rejected(self):
        payload = {**self.payload, "num_frames": 141}
        with self.assertRaisesRegex(client.ClientError, "wrong frame count"):
            client.validate_media(self.media, payload)


if __name__ == "__main__":
    unittest.main()
