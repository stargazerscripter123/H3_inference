#!/usr/bin/env python3
"""Prepare or validate an Omni AdaLN table using the pinned FastVideo driver.

The prepare mode keeps one SP1 generator alive for all requests. It publishes
the table only after every request and the table validation pass. The validate
mode creates a fresh generator using only the published table and supports SP2.
See prepare_omni_adaln.md for the request-file contract and examples.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from typing import Any


FASTVIDEO_REVISION = "cdda1b7535fe32699335e39db82c1d4db21ceb88"
QAD_REVISION = "f01e6a23ac6066aa9eb622cf94b6259442e70ec2"
DRIVER = "examples/inference/basic/basic_fasth3_omniref_pdd.py"
COMPONENTS = (
    "transformer_ref", "text_encoder", "tokenizer", "processor", "vae",
    "audio_vae", "scheduler", "audio_scheduler",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_record(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_requests(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"requests"}:
        raise ValueError("Request file must contain exactly a 'requests' array")
    requests = raw["requests"]
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a nonempty array")
    seen: set[str] = set()
    normalized = []
    allowed = {"id", "prompt", "references", "num_frames", "seed", "phase"}
    for item in requests:
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError(f"Invalid request or unknown fields: {item!r}")
        name = item.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
            raise ValueError("Each request needs a safe alphanumeric id (up to 80 characters)")
        if name in seen:
            raise ValueError(f"Duplicate request id: {name}")
        seen.add(name)
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            raise ValueError(f"{name}: prompt must be a nonempty string")
        frames = item.get("num_frames", 124)
        if type(frames) is not int or frames not in (124, 362):
            raise ValueError(f"{name}: num_frames must be 124 or 362")
        seed = item.get("seed", 1234)
        if type(seed) is not int or seed < 0:
            raise ValueError(f"{name}: seed must be a nonnegative integer")
        phase = item.get("phase", "smoke")
        if phase not in ("prepare", "smoke", "warmup", "timed"):
            raise ValueError(f"{name}: unsupported phase {phase!r}")
        refs = item.get("references")
        if not isinstance(refs, list) or not refs or len(refs) > 12:
            raise ValueError(f"{name}: supply 1 to 12 ordered references")
        counts = dict(image=0, video=0, audio=0)
        resolved_refs = []
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {"media_type", "source"}:
                raise ValueError(f"{name}: reference fields must be media_type and source")
            kind = ref["media_type"]
            if kind not in counts or not isinstance(ref["source"], str):
                raise ValueError(f"{name}: invalid reference {ref!r}")
            source = Path(ref["source"]).expanduser()
            if not source.is_absolute():
                source = path.parent / source
            source = source.resolve(strict=True)
            if not source.is_file():
                raise ValueError(f"{name}: reference is not a regular file: {source}")
            counts[kind] += 1
            resolved_refs.append({"media_type": kind, "source": str(source)})
        if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
            raise ValueError(f"{name}: reference count exceeds the release contract")
        if not counts["image"] + counts["video"]:
            raise ValueError(f"{name}: Ref2VA requires at least one image or video reference")
        normalized.append(dict(id=name, prompt=item["prompt"], references=resolved_refs,
                               num_frames=frames, seed=seed, phase=phase))
    return normalized


def inspect_table(path: Path, torch: Any, hidden_size: int = 5376) -> dict[str, Any]:
    """Fail closed on partial dumps, inconsistent clocks, bad shapes or NaNs."""
    table = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(table, dict) or set(table) != set(range(50)):
        raise ValueError("AdaLN table must contain exactly blocks 0 through 49")
    keys: set[str] | None = None
    descriptions = []
    tensor_count = 0
    for index in range(50):
        block = table[index]
        if not isinstance(block, dict) or len(block) < 8:
            raise ValueError(f"Block {index} has fewer than eight clock entries")
        if keys is None:
            keys = set(block)
        elif set(block) != keys:
            raise ValueError(f"Block {index} has different clock entries")
        for key, tensor in block.items():
            if not isinstance(key, str):
                raise ValueError(f"Block {index}: clock key must be a string")
            parsed = ast.literal_eval(key)
            if (not isinstance(parsed, tuple) or len(parsed) != 3
                    or not isinstance(parsed[0], tuple) or not parsed[0]
                    or not isinstance(parsed[1], tuple) or len(parsed[1]) != 2
                    or not isinstance(parsed[2], str)):
                raise ValueError(f"Block {index}: malformed clock key {key!r}")
            clock, embedding_shape, dtype = parsed
            if any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in clock):
                raise ValueError(f"Block {index}: non-finite clock")
            if embedding_shape[0] != len(clock):
                raise ValueError(f"Block {index}: clock/embedding dimensions disagree")
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                raise ValueError(f"Block {index}: modulation must be a floating tensor")
            if tensor.ndim != 2 or tuple(tensor.shape) != (len(clock), 18 * hidden_size):
                raise ValueError(f"Block {index}: invalid modulation tensor shape")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"Block {index}: non-finite modulation tensor")
            # The key describes time_embedder output (FP32 in this release),
            # while the projection casts to its own weight dtype (BF16).
            # They are intentionally different in a valid native QAD dump.
            if dtype not in ("torch.float32", "torch.bfloat16", "torch.float16"):
                raise ValueError(f"Block {index}: unsupported embedding dtype")
            if str(tensor.dtype) not in ("torch.float32", "torch.bfloat16", "torch.float16"):
                raise ValueError(f"Block {index}: unsupported modulation dtype")
            tensor_count += 1
            if index == 0:
                descriptions.append({"key": key, "shape": list(tensor.shape),
                                     "embedding_dtype": dtype, "modulation_dtype": str(tensor.dtype)})
    return {"blocks": 50, "keys_per_block": len(keys or ()), "tensors": tensor_count,
            "bytes": path.stat().st_size, "sha256": sha256(path),
            "clock_entries": sorted(descriptions, key=lambda entry: entry["key"])}


def inspect_media(path: Path, expected_frames: int, width: int, height: int) -> dict[str, Any]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, text=True, capture_output=True,
    )
    info = json.loads(proc.stdout)
    videos = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    audios = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ValueError("Expected one video and one audio stream")
    video, audio = videos[0], audios[0]
    frames = int(video.get("nb_read_frames", video.get("nb_frames", -1)))
    fps = float(Fraction(video["avg_frame_rate"]))
    if (video.get("width"), video.get("height"), frames) != (width, height, expected_frames):
        raise ValueError(f"Unexpected output geometry/frame count: {video}")
    if not math.isclose(fps, 24.0, abs_tol=0.01):
        raise ValueError(f"Unexpected output fps: {fps}")
    if int(audio.get("sample_rate", 0)) != 32000 or int(audio.get("channels", 0)) != 2:
        raise ValueError(f"Expected 32-kHz stereo output: {audio}")
    video_duration = float(video.get("duration", frames / fps))
    audio_duration = float(audio.get("duration", info["format"]["duration"]))
    if not math.isfinite(audio_duration) or abs(audio_duration - video_duration) > 0.20:
        raise ValueError(f"A/V durations differ: video={video_duration}, audio={audio_duration}")
    return {"width": width, "height": height, "frames": frames, "fps": fps,
            "sample_rate": 32000, "channels": 2, "video_duration": video_duration,
            "audio_duration": audio_duration, "bytes": path.stat().st_size}


def import_driver(root: Path) -> Any:
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          text=True, capture_output=True).stdout.strip()
    if head != FASTVIDEO_REVISION:
        raise ValueError(f"FastVideo must be pinned to {FASTVIDEO_REVISION}; found {head}")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", DRIVER],
                   check=True, capture_output=True)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("_pinned_fasth3_omni_driver", root / DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the pinned FastVideo Omni driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "validate"), default="prepare")
    parser.add_argument("--fastvideo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Complete local QAD snapshot")
    parser.add_argument("--decoder", type=Path, required=True, help="Packed 36L or 8L NVFP4 decoder")
    parser.add_argument("--requests-json", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist")
    parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.mode == "prepare" and args.num_gpus != 1:
        parser.error("Prepare with SP1 to avoid duplicate host projection weights")
    for name in ("fastvideo_root", "model_dir", "decoder", "requests_json"):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    args.table = args.table.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    requests = read_requests(args.requests_json)
    if args.mode == "prepare":
        has_image_only = any(all(ref["media_type"] == "image" for ref in item["references"])
                             for item in requests)
        has_audio_reference = any(any(ref["media_type"] == "audio" for ref in item["references"])
                                  for item in requests)
        if not has_image_only or not has_audio_reference:
            parser.error("Prepare requires both an image-only request and a request with an explicit audio reference; "
                         "their unique timestep sets can differ")
    for component in COMPONENTS:
        if not (args.model_dir / component).is_dir():
            parser.error(f"Complete QAD snapshot is missing {component}/")
    if not (args.model_dir / "transformer_ref/nvfp4_weights.safetensors").is_file():
        parser.error("Model is missing packed transformer_ref/nvfp4_weights.safetensors")
    architecture = json.loads((args.model_dir / "transformer_ref/config.json").read_text())
    hidden_size = architecture.get("hidden_size")
    if type(hidden_size) is not int or hidden_size <= 0 or architecture.get("num_layers") != 50:
        parser.error("Expected the released 50-block transformer architecture with a positive hidden_size")
    if not args.decoder.is_file():
        parser.error("--decoder must be a local file")
    if shutil.which("ffprobe") is None:
        parser.error("ffprobe is required for complete audio/video validation")
    if args.mode == "prepare" and (args.table.exists() or Path(str(args.table) + ".inputs").exists()):
        parser.error("Refusing to replace an existing AdaLN table or companion inputs")
    if args.mode == "validate" and not args.table.is_file():
        parser.error("--mode validate requires an existing table")
    if os.environ.get("FASTVIDEO_H3_VSA_FINE_ATTENTION"):
        parser.error("Unset FASTVIDEO_H3_VSA_FINE_ATTENTION for the native release path")

    # Set before importing FastVideo: environment-backed settings and spawned
    # workers must observe the intended mode from their first import.
    os.environ.pop("FASTVIDEO_H3_ADALN_TABLE", None)
    os.environ.pop("FASTVIDEO_H3_ADALN_DUMP", None)
    os.environ.pop("FASTVIDEO_H3_ADALN_CACHE", None)
    partial_table = args.output_dir / "adaln_table.partial.pt"
    if args.mode == "prepare":
        os.environ["FASTVIDEO_H3_ADALN_CACHE"] = "1"
        os.environ["FASTVIDEO_H3_ADALN_DUMP"] = str(partial_table)
    else:
        os.environ["FASTVIDEO_H3_ADALN_TABLE"] = str(args.table)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records_path = args.output_dir / "requests.jsonl"
    driver = import_driver(args.fastvideo_root)
    import torch

    first = requests[0]
    driver_argv = ["--model-path", str(args.model_dir), "--base-model-path", str(args.model_dir),
                   "--composed-dir", str(args.output_dir / "composed"), "--profile", "rtx-5090",
                   "--output", str(args.output_dir), "--prompt", first["prompt"]]
    for ref in first["references"]:
        driver_argv.extend(["--" + ref["media_type"], ref["source"]])
    model_dir, contract = driver.resolve_model(driver.parse_args(driver_argv))
    if (contract.get("num_inference_steps") != 8 or contract.get("transformer_forwards") != 8
            or contract.get("pdd_steps") != 32 or contract.get("model_type") != "ref2va"
            or contract.get("pdd_step_indices") != list(range(0, 33, 4))
            or contract.get("video_scheduler_shift") != 12.0
            or contract.get("audio_scheduler_shift") != 3.0):
        raise ValueError("Checkpoint does not match the released Omni eight-forward contract")
    profile = driver.DEVICE_PROFILES["rtx-5090"]
    for key, value in profile.env.items():
        os.environ[key] = value
    os.environ["FASTVIDEO_H3_VAE_NVFP4_DECODER"] = str(args.decoder)
    driver.apply_profile(profile, str(args.decoder), str(args.table) if args.mode == "validate" else None)
    manifest = {"mode": args.mode, "status": "starting", "fastvideo_revision": FASTVIDEO_REVISION,
                "expected_qad_revision": QAD_REVISION, "model_revision_verification": "external download manifest required",
                "model_dir": str(args.model_dir), "model_contract_sha256": sha256(args.model_dir / driver.CONTRACT),
                "decoder": str(args.decoder), "requests_json_sha256": sha256(args.requests_json),
                "num_gpus": args.num_gpus, "tp_size": 1, "sp_size": args.num_gpus, "fsdp": False,
                "configured_transformer_forwards": 8, "measured_transformer_forwards": None,
                "frame_geometry": [832, 480], "fps": 24, "contract": contract,
                "requests": requests, "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
                "environment": {key: value for key, value in os.environ.items() if key.startswith("FASTVIDEO_")}}
    if args.mode == "validate":
        manifest["input_table"] = inspect_table(args.table, torch, hidden_size)
    manifest_path = args.output_dir / "run.json"
    write_json(manifest_path, manifest)
    generator = None
    start = time.perf_counter()
    try:
        generator = driver.VideoGenerator.from_config(
            driver.build_generator_config(model_dir, args.num_gpus, False, profile))
        manifest["generator_startup_seconds"] = time.perf_counter() - start
        write_json(manifest_path, manifest)
        for index, item in enumerate(requests):
            output_path = args.output_dir / f"{index:03d}_{item['id']}.mp4"
            request = driver.GenerationRequest(
                prompt=item["prompt"], negative_prompt="",
                inputs=driver.InputConfig(references=[driver.MiniMaxH3Reference(**ref) for ref in item["references"]]),
                sampling=driver.SamplingConfig(height=480, width=832, num_frames=item["num_frames"],
                                               fps=24, num_inference_steps=8, guidance_scale=1.0,
                                               batch_cfg=False, seed=item["seed"]),
                output=driver.OutputConfig(output_path=str(output_path), save_video=True, return_frames=False),
            )
            record = {"id": item["id"], "phase": item["phase"], "seed": item["seed"],
                      "num_frames": item["num_frames"], "configured_transformer_forwards": 8,
                      "measured_transformer_forwards": None, "output": str(output_path)}
            request_start = time.perf_counter()
            try:
                result = generator.generate(request)
                record["generation_wall_seconds"] = time.perf_counter() - request_start
                record["reported_generation_seconds"] = result.generation_time
                record["reported_rank0_peak_allocated_mb"] = result.peak_memory_mb
                record["per_rank_peak_allocated_mb"] = None
                actual_output = Path(result.video_path) if result.video_path else output_path
                if not actual_output.is_file():
                    raise RuntimeError("Generation returned without a complete local video file")
                record["output"] = str(actual_output.resolve())
                check_start = time.perf_counter()
                record["media"] = inspect_media(actual_output, item["num_frames"], 832, 480)
                record["media_validation_seconds"] = time.perf_counter() - check_start
                if args.mode == "prepare":
                    record["table_progress"] = inspect_table(partial_table, torch, hidden_size)
                record["status"] = "passed"
                append_record(records_path, record)
                print(json.dumps({key: record[key] for key in ("id", "status", "generation_wall_seconds", "output")}), flush=True)
            except BaseException as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}",
                              elapsed_until_failure_seconds=time.perf_counter() - request_start)
                append_record(records_path, record)
                raise
        if args.mode == "prepare":
            verified = inspect_table(partial_table, torch, hidden_size)
            args.table.parent.mkdir(parents=True, exist_ok=True)
            # link fails atomically if another process has published this name;
            # copy-to-sibling keeps publication on the destination filesystem.
            staging = args.table.with_name(args.table.name + f".publish-{os.getpid()}")
            try:
                with staging.open("xb") as destination, partial_table.open("rb") as source:
                    shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                os.link(staging, args.table)
            finally:
                staging.unlink(missing_ok=True)
            manifest["published_table"] = {"path": str(args.table), **verified}
            # The .inputs file is diagnostic only and stays beside the partial
            # table. Do not turn it into an undocumented inference dependency.
        else:
            manifest["verified_table"] = inspect_table(args.table, torch, hidden_size)
        manifest["status"] = "passed"
        manifest["total_wall_seconds"] = time.perf_counter() - start
        write_json(manifest_path, manifest)
        return 0
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        total_wall_seconds=time.perf_counter() - start)
        write_json(manifest_path, manifest)
        raise
    finally:
        if generator is not None:
            generator.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
