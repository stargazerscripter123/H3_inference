"""Versioned upstream patch tests using source ASTs and geometry-only stubs.

No PIL, NumPy, Torch, media files or model weights are imported/read. The patch
is applied only to temporary copies of its four Python sources. Set
OMNI_FASTVIDEO_SRC to test against an original or already-patched pinned tree.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("OMNI_FASTVIDEO_SRC", str(PACKAGE.parent / "upstream/FastVideo")))
CONFIG = "fastvideo/configs/pipelines/minimax_h3.py"
REFERENCE = "fastvideo/pipelines/basic/minimax_h3/reference.py"
INPUT = "fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_input_preparation.py"
CONDITIONING = "fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_conditioning.py"
PACKING = "fastvideo/pipelines/basic/minimax_h3/packing.py"
CAP_FIELD = "h3_reference_image_max_short_edge"
POLICY_FIELD = "h3_reference_image_resize_policy"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def nodes(source, names):
    return [copy.deepcopy(node) for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]


def execute(selected, namespace):
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                            *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<upstream-source-ast>", "exec"), namespace)
    return namespace


class ShapeImage:
    """Dimension record only; contains no pixel data and opens no media."""
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.size = (width, height)

    def convert(self, mode):
        assert mode == "RGB"
        return self

    def resize(self, size, resample):
        return ShapeImage(*size)


class Prepared:
    def __init__(self, media_type="image", image=None, frames=None):
        self.media_type, self.image, self.frames = media_type, image, frames
        self.has_audio, self.block_timestamps = False, []


class Memo:
    enabled = True

    def __init__(self):
        self.values = {}

    def get_or_compute(self, key, compute):
        if key not in self.values:
            self.values[key] = compute()
        return self.values[key]


class ReferenceImageCapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = json.loads((PACKAGE / "patches/reference-image-cap-v1.json").read_text())
        patch = (PACKAGE / "patches/reference-image-cap-v1.patch").read_bytes()
        if sha(patch) != manifest["patch_sha256"]:
            raise AssertionError("Patch SHA256 does not match version manifest")
        modes = set()
        cls.work = tempfile.TemporaryDirectory(prefix="omni-refcap-test-")
        cls.addClassCleanup(cls.work.cleanup)
        root = Path(cls.work.name)
        for relative, hashes in manifest["files"].items():
            raw = (SOURCE / relative).read_bytes()
            mode = "before" if sha(raw) == hashes["before_sha256"] else "after" if sha(raw) == hashes["after_sha256"] else None
            if mode is None:
                raise AssertionError("Source hash is neither pinned original nor patched: " + relative)
            modes.add(mode)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        if len(modes) != 1:
            raise AssertionError("Source tree is partially patched")

        def apply(*flags):
            subprocess.run(["patch", "-p1", "--batch", *flags], cwd=root, input=patch,
                           capture_output=True, check=True)

        if modes == {"after"}:
            apply("-R")
        cls.before = {relative: (root / relative).read_text() for relative in manifest["files"]}
        if any(sha(text.encode()) != manifest["files"][relative]["before_sha256"]
               for relative, text in cls.before.items()):
            raise AssertionError("Reverse patch does not recover exact pinned originals")
        apply("--dry-run")
        apply()
        cls.after = {relative: (root / relative).read_text() for relative in manifest["files"]}
        if any(sha(text.encode()) != manifest["files"][relative]["after_sha256"]
               for relative, text in cls.after.items()):
            raise AssertionError("Applied patch does not reproduce expected hashes")
        for relative, text in cls.after.items():
            compile(text, relative, "exec")
        packing = ast.parse((SOURCE / PACKING).read_text())
        cls.multiple = next(ast.literal_eval(n.value) for n in packing.body
                            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                            and t.id == "MINIMAX_H3_CANVAS_MULTIPLE" for t in n.targets))

    def geometry(self, original=False):
        ns = {"MINIMAX_H3_CANVAS_MULTIPLE": self.multiple, "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE": 2048}
        text = self.before[REFERENCE] if original else self.after[REFERENCE]
        return execute(nodes(text, {"resolve_reference_image_size"}), ns)["resolve_reference_image_size"]

    def test_none_preserves_pinned_native_geometry_exactly(self):
        native, changed = self.geometry(True), self.geometry()
        sizes = [(1, 1), (4, 1), (31, 32), (188, 47), (864, 480), (1920, 1080), (480, 864), (8192, 2048)]
        for width in range(32, 2050, 47):
            sizes.extend((width, height) for height in range(32, 2050, 113)
                         if max(width, height) <= 4 * min(width, height))
        for size in sizes:
            self.assertEqual(changed(*size), native(*size))
            self.assertEqual(changed(*size, None), native(*size))

    def test_cap480_examples_and_small_image_aspect_boundary(self):
        expected = {(1920, 1080): (480, 832), (1080, 1920): (832, 480),
                    (864, 480): (480, 864), (640, 360): (352, 608),
                    (188, 47): (32, 128), (32, 32): (32, 32), (1000, 1000): (480, 480)}
        self.assertEqual(self.multiple, 32)
        for source, result in expected.items():
            with self.subTest(source=source):
                self.assertEqual(self.geometry()(*source, 480, "cap"), result)

    def test_capped_geometry_properties_no_upscale_and_aspect_bound(self):
        resolve = self.geometry()
        for width in range(32, 4097, 61):
            for height in range(32, 4097, 137):
                if max(width, height) > 4 * min(width, height):
                    continue
                for cap in (32, 479, 480, 481, 720):
                    out_h, out_w = resolve(width, height, cap, "cap")
                    self.assertTrue(32 <= out_h <= height and 32 <= out_w <= width)
                    self.assertEqual((out_h % 32, out_w % 32), (0, 0))
                    self.assertLessEqual(min(out_h, out_w), cap)
                    self.assertLessEqual(max(out_h, out_w), 4 * min(out_h, out_w))
                    ideal_long = max(width, height) * min(out_h, out_w) / min(width, height)
                    self.assertTrue(0 <= ideal_long - max(out_h, out_w) < 32)

    def test_invalid_cap_and_too_small_or_invalid_aspect_rejected(self):
        resolve = self.geometry()
        for cap in (True, False, 0, -1, 31, 480.0, "480", float("nan")):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                resolve(864, 480, cap, "cap")
        for source in ((31, 32), (32, 31), (0, 32), (32, 129)):
            with self.subTest(source=source), self.assertRaises(ValueError):
                resolve(*source, 480, "cap")

    def test_cap720_and_original_are_distinct_from_native(self):
        resolve = self.geometry()
        self.assertEqual(resolve(1920, 1080, 720, "cap"), (704, 1248))
        self.assertEqual(resolve(1920, 1080, None, "original"), (1056, 1856))
        self.assertEqual(resolve(864, 480, 720, "cap"), (480, 864))
        self.assertEqual(resolve(864, 480, None, "original"), (480, 864))
        self.assertEqual(resolve(188, 47, None, "original"), (32, 128))
        for width in range(32, 4097, 61):
            for height in range(32, 4097, 137):
                if max(width, height) > 4 * min(width, height):
                    continue
                out_h, out_w = resolve(width, height, None, "original")
                self.assertTrue(32 <= out_h <= height and 32 <= out_w <= width)
                self.assertEqual((out_h % 32, out_w % 32), (0, 0))
                self.assertEqual(min(out_h, out_w), min(width, height) // 32 * 32)
                self.assertLessEqual(max(out_h, out_w), 4 * min(out_h, out_w))
                self.assertTrue(0 <= max(width, height) * min(out_h, out_w) / min(width, height)
                                - max(out_h, out_w) < 32)

    def test_config_default_and_strict_validation(self):
        class Base:
            def check_pipeline_config(self):
                pass
        ns = {"__name__": __name__, "dataclass": dataclass, "field": field, "math": math, "PipelineConfig": Base,
              **{name: NS for name in ("MiniMaxH3Config", "MiniMaxH3VideoVAEConfig", "MiniMaxH3AudioVAEConfig", "MiniMaxH3Qwen3VLConfig")}}
        config = execute(nodes(self.after[CONFIG], {"MiniMaxH3PipelineConfig"}), ns)["MiniMaxH3PipelineConfig"]
        self.assertIsNone(getattr(config(), CAP_FIELD))
        self.assertEqual(getattr(config(), POLICY_FIELD), "native")
        for policy, cap in (("native", None), ("original", None), ("cap", 32), ("cap", 479), ("cap", 480), ("cap", 720)):
            config(**{CAP_FIELD: cap, POLICY_FIELD: policy}).check_pipeline_config()
        for value in (None, True, False, "480", 480.0, 31, 0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, CAP_FIELD):
                config(**{CAP_FIELD: value, POLICY_FIELD: "cap"}).check_pipeline_config()
        for policy, cap in (("native", 480), ("original", 480), ("original", 0), ("native", 0), ("unknown", None), (None, None), (True, None)):
            with self.subTest(policy=policy, cap=cap), self.assertRaises(ValueError):
                config(**{CAP_FIELD: cap, POLICY_FIELD: policy}).check_pipeline_config()
            with self.subTest(policy=policy, cap=cap), self.assertRaises(ValueError):
                self.geometry()(864, 480, cap, policy)
        with self.assertRaises(ValueError):
            self.geometry()(31, 32, None, "original")

    def test_reference_preparation_uses_geometry_and_logs_numbers_only(self):
        logger = NS(info=Mock())
        ns = {"os": os, "Image": NS(Image=ShapeImage, Resampling=NS(LANCZOS=1)),
              "ImageOps": NS(exif_transpose=lambda image: image), "MiniMaxH3PreparedReference": Prepared,
              "MINIMAX_H3_CANVAS_MULTIPLE": 32, "resolve_reference_image_size": self.geometry(), "logger": logger,
              "load_image": Mock(side_effect=AssertionError("Media file access is forbidden"))}
        execute(nodes(self.after[REFERENCE], {"prepare_reference_image", "prepare_reference"}), ns)
        reference = NS(media_type="image", source=ShapeImage(1920, 1080))
        result = ns["prepare_reference"](reference, 362, 32000, image_max_short_edge=480, image_resize_policy="cap")
        self.assertEqual(result.image.size, (832, 480))
        ns["load_image"].assert_not_called()
        self.assertEqual(logger.info.call_args.args[1:], ("cap", 1920, 1080, 832, 480, 480, 32))

    def input_class(self, prepare):
        ns = {"PipelineStage": object, "torch": NS(no_grad=lambda: lambda method: method),
              "prepare_common_request": lambda batch: None, "validate_references": lambda refs: refs,
              "prepare_reference": prepare, "resolve_target_canvas": lambda batch, *_: (batch.height, batch.width, 16),
              "resolve_target_num_frames": lambda frames: frames, "video_latent_num_frames": lambda frames: 43,
              "MINIMAX_H3_KEYFRAMES_KEY": "keyframes", "MINIMAX_H3_KEYFRAME_ANCHORS_KEY": "anchors"}
        return execute(nodes(self.after[INPUT], {"MiniMaxH3InputPreparationStage"}), ns)["MiniMaxH3InputPreparationStage"]

    def test_input_stage_propagates_config_without_changing_output_contract(self):
        for policy, cap in (("native", None), ("cap", 480), ("cap", 720), ("original", None)):
            prepare = Mock(side_effect=lambda reference, *args, **kwargs: reference)
            stage = self.input_class(prepare)(NS(spatial_compression_ratio=16, latent_channels=16),
                                              NS(sampling_rate=32000), ref2va=True)
            refs = [Prepared("image"), Prepared("video"), Prepared("audio")]
            batch = NS(image_path=None, pil_image=None, last_image=None, references=refs,
                       width=832, height=480, num_frames=362, fps=24, extra={})
            stage.forward(batch, NS(pipeline_config=NS(**{CAP_FIELD: cap, POLICY_FIELD: policy})))
            self.assertEqual((batch.width, batch.height, batch.num_frames, batch.fps), (832, 480, 362, 24))
            self.assertEqual(prepare.call_count, 3)
            for call in prepare.call_args_list:
                self.assertEqual(call.args[1:], (362, 32000))
                self.assertEqual(call.kwargs, {"image_max_short_edge": cap, "image_resize_policy": policy})

    def conditioning_class(self):
        ns = {"PipelineStage": object, "torch": NS(no_grad=lambda: lambda method: method),
              "MiniMaxH3PreparedReference": Prepared, "image_key": lambda image: image.size,
              "get_local_torch_device": lambda: "cpu", "DTensor": type("DTensor", (), {}),
              "nvtx_range": lambda name: nullcontext(), "MINIMAX_H3_TEXT_TOKEN_TAGS_KEY": "tags",
              "build_ref2va_presentation": lambda *args: ([1], [2]), "np": NS(stack=lambda frames: frames),
              "sample_reference_video_frames": lambda frames: (["geometry-only-frame"], [0.0])}
        return execute(nodes(self.after[CONDITIONING], {"MiniMaxH3ConditioningStage"}), ns)["MiniMaxH3ConditioningStage"]

    def test_image_processor_no_resize_only_when_capped_video_policy_unchanged(self):
        for policy, cap in (("native", None), ("cap", 480), ("cap", 720), ("original", None)):
            image_processor = Mock(return_value={"pixel_values": "image-tokens",
                                   "image_grid_thw": [NS(prod=lambda: NS(item=lambda: 16))]})
            image_processor.merge_size = 2
            video_processor = Mock(return_value={"pixel_values_videos": "video-tokens", "video_grid_thw": [(1, 30, 52)]})
            stage = object.__new__(self.conditioning_class())
            stage.processor = NS(image_processor=image_processor, video_processor=video_processor)
            stage.tokenizer, stage._encode_tokens = None, Mock(return_value=("embeds", "tags"))
            image = ShapeImage(128, 32)
            refs = [Prepared(image=image), Prepared("video", frames="geometry-only-frames")]
            stage._present_ref2va(NS(prompt="synthetic prompt"), refs, "cpu", policy)
            expected = {"images": [image], "return_tensors": "pt"}
            if policy != "native":
                expected["do_resize"] = False
            image_processor.assert_called_once_with(**expected)
            self.assertEqual(video_processor.call_args.kwargs,
                             {"videos": [["geometry-only-frame"]], "do_sample_frames": False, "return_tensors": "pt"})

    def test_conditioning_memo_separates_resize_policy_for_identical_geometry(self):
        stage = object.__new__(self.conditioning_class())
        stage._ref2va_memo = Memo()
        stage._present_ref2va = Mock(side_effect=lambda *args: args[-1])
        batch = NS(prompt="synthetic prompt", references=[Prepared(image=ShapeImage(128, 32))])
        for policy in ("native", "cap", "original"):
            self.assertEqual(stage._encode_ref2va(batch, "cpu", policy), policy)
            self.assertEqual(stage._encode_ref2va(batch, "cpu", policy), policy)
        self.assertEqual(stage._present_ref2va.call_count, 3)
        self.assertEqual(len(stage._ref2va_memo.values), 3)

    def test_conditioning_forward_propagates_config(self):
        for policy, cap in (("native", None), ("cap", 480), ("cap", 720), ("original", None)):
            stage = object.__new__(self.conditioning_class())
            stage.conditioner = NS(parameters=lambda: iter(()))
            stage.ref2va, stage._encode_ref2va = True, Mock(return_value=("embeds", "tags"))
            batch = NS(extra={})
            args = NS(text_encoder_cpu_offload=False, pipeline_config=NS(**{CAP_FIELD: cap, POLICY_FIELD: policy}))
            stage.forward(batch, args)
            stage._encode_ref2va.assert_called_once_with(batch, "cpu", policy)
            self.assertEqual(batch.prompt_embeds, ["embeds"])


if __name__ == "__main__":
    unittest.main()
