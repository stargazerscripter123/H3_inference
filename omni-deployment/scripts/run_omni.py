#!/usr/bin/env python3
"""FastH3 Omni client for the pinned FastVideo OpenAI video-job API.

Run beside the server when using local references. For references already on a
remote server, use --server-paths with absolute paths visible to that server.
This client never uploads or preprocesses references. Order is image(s), then
video(s), then audio(s), preserving the order within each type; the pinned HTTP
adapter cannot represent arbitrary interleaving. An invalid order is rejected.

Contract source (commit cdda1b7535fe32699335e39db82c1d4db21ceb88):
  fastvideo/entrypoints/openai/{protocol,request_adapter,video_api}.py
  examples/inference/basic/basic_fasth3_omniref_rtx5090.yaml
The Omni PDD recipe uses num_inference_steps=8 (eight forwards), without the
legacy FL2VA client's +1 conversion. No requests are automatically resubmitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import uuid

FASTVIDEO_REVISION = "cdda1b7535fe32699335e39db82c1d4db21ceb88"
ORDER = {"image": 0, "video": 1, "audio": 2}
LIMITS = {"image": 9, "video": 3, "audio": 3}


class ClientError(RuntimeError):
    def __init__(self, message: str, *, manifest: dict[str, Any] | None = None):
        super().__init__(message)
        self.manifest = manifest


class HTTPFailure(ClientError):
    def __init__(self, status: int, path: str, detail: str):
        super().__init__(f"HTTP {status} from {path}: {detail}")
        self.status = status


class ReferenceAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        refs = list(getattr(namespace, self.dest, None) or [])
        refs.append((self.const, values))
        setattr(namespace, self.dest, refs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def normalize_server(server: str) -> str:
    parsed = urllib.parse.urlsplit(server)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") not in {"", "/v1"}):
        raise ClientError("--server must be an HTTP(S) origin, optionally ending in /v1, without credentials")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def require_positive_finite(**values: float) -> None:
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ClientError(f"{name} must be a positive finite number")


def build_request(prompt: str, references: list[tuple[str, str]], *, seed: int,
                  frames: int, width: int, height: int, model: str | None,
                  server_paths: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ClientError("prompt must not be empty")
    for name, value in (("seed", seed), ("frames", frames), ("width", width), ("height", height)):
        if type(value) is not int:
            raise ClientError(f"{name} must be an integer (booleans and floats are not accepted)")
    if not 124 <= frames <= 362 or frames % 17 != 5:
        raise ClientError("--frames must be 124..362 on the 17*n+5 grid (for example 124 or 362)")
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ClientError("width and height must be positive multiples of 32")
    if not -(2**63) <= seed < 2**63:
        raise ClientError("seed must fit a signed 64-bit integer")
    if not references or len(references) > 12:
        raise ClientError("Omni requires 1..12 references")
    counts = {kind: 0 for kind in ORDER}
    ranks = []
    for reference in references:
        if (not isinstance(reference, (tuple, list)) or len(reference) != 2
                or not all(isinstance(part, str) for part in reference)):
            raise ClientError("each reference must contain a string type and source path")
        kind, source = reference
        if kind not in ORDER:
            raise ClientError(f"unsupported reference type: {kind}")
        if not source or "\x00" in source:
            raise ClientError("reference path must not be empty or contain NUL")
        counts[kind] += 1
        ranks.append(ORDER[kind])
    if ranks != sorted(ranks):
        raise ClientError("the pinned HTTP API requires image(s), video(s), audio(s) order; interleaving would be silently reordered")
    if counts["image"] + counts["video"] == 0:
        raise ClientError("audio references require at least one image or video")
    for kind, limit in LIMITS.items():
        if counts[kind] > limit:
            raise ClientError(f"at most {limit} {kind} references are supported")
    payload: dict[str, Any] = {
        "task": "ref2va", "prompt": prompt, "seed": seed,
        "width": width, "height": height, "num_frames": frames, "fps": 24,
        "num_inference_steps": 8, "guidance_scale": 1.0,
        "num_outputs_per_prompt": 1,
    }
    if model:
        payload["model"] = model
    audit = []
    per_type = {kind: 0 for kind in ORDER}
    for index, (kind, source) in enumerate(references):
        if server_paths:
            # Do not resolve server paths against this client's home or cwd.
            if not source.startswith("/") or "\x00" in source:
                raise ClientError("--server-paths requires absolute paths on the server")
            resolved = source
            info: dict[str, Any] = {"sha256": None, "bytes": None, "verified_on_client": False}
        else:
            path = Path(source).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ClientError(f"reference is not a regular file: {path}")
            resolved = str(path)
            info = {"sha256": sha256_file(path), "bytes": path.stat().st_size,
                    "verified_on_client": True}
        payload.setdefault(f"{kind}_reference", []).append({f"{kind}_url": resolved})
        per_type[kind] += 1
        audit.append({"index": index, "type": kind, "index_within_type": per_type[kind],
                      "source": resolved, **info})
    return payload, audit


def read_json(server: str, path: str, *, timeout: float,
              payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(server + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode(errors="replace")
        raise HTTPFailure(error.code, path, detail) from error
    if not isinstance(result, dict):
        raise ClientError(f"expected an object from {path}")
    return result


def validate_media(path: Path, payload: dict[str, Any], *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    command = [ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format",
               "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    if result.stderr.strip():
        raise ClientError(f"ffprobe decoding errors: {result.stderr[:1500]}")
    probe = json.loads(result.stdout)
    video = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
    audio = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"]
    if len(video) != 1 or len(audio) != 1:
        raise ClientError(f"expected one video and one audio stream, got {len(video)} and {len(audio)}")
    v, a = video[0], audio[0]
    try:
        count = int(v["nb_read_frames"])
        fps = float(Fraction(v["avg_frame_rate"]))
        video_s, audio_s = float(v["duration"]), float(a["duration"])
        sample_rate, channels = int(a["sample_rate"]), int(a["channels"])
        audio_frames = int(a["nb_read_frames"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ClientError("ffprobe did not return complete video/audio metadata") from error
    expected_s = payload["num_frames"] / payload["fps"]
    errors = []
    if (v.get("width"), v.get("height")) != (payload["width"], payload["height"]):
        errors.append("wrong resolution")
    if count != payload["num_frames"]:
        errors.append(f"wrong frame count: {count}")
    if abs(fps - payload["fps"]) > 1e-6:
        errors.append(f"wrong fps: {fps}")
    if sample_rate != 32000 or channels != 2 or audio_frames < 1:
        errors.append(f"wrong audio format: {sample_rate} Hz, {channels} channels, {audio_frames} frames")
    if (not all(math.isfinite(x) for x in (video_s, audio_s, fps))
            or abs(video_s - expected_s) > 0.1 or abs(audio_s - expected_s) > 0.25):
        errors.append(f"wrong durations: video={video_s}, audio={audio_s}, expected={expected_s}")
    if errors:
        raise ClientError("invalid generated media: " + "; ".join(errors))
    return {"valid": True, "width": v["width"], "height": v["height"], "frames": count,
            "fps": fps, "video_duration_s": video_s, "audio_duration_s": audio_s,
            "audio_sample_rate": sample_rate, "audio_channels": channels,
            "bytes": path.stat().st_size, "sha256": sha256_file(path), "ffprobe": probe}


def run_job(server: str, payload: dict[str, Any], references: list[dict[str, Any]], *,
            output_root: Path, timeout: float = 1800, poll_interval: float = 0.25,
            http_timeout: float = 60, ffprobe: str = "ffprobe",
            cache_condition: str = "unspecified") -> dict[str, Any]:
    require_positive_finite(timeout=timeout, poll_interval=poll_interval, http_timeout=http_timeout)
    server = normalize_server(server)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = output_root.expanduser().resolve() / f"omni_{stamp}_{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    manifest_path = directory / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1, "client_contract_revision": FASTVIDEO_REVISION,
        "server": server, "created_at": datetime.now(timezone.utc).isoformat(),
        "request": payload, "references": references, "cache_condition": cache_condition,
        "cache_condition_note": "operator label; the API does not expose cache hit verification",
        "reference_order": "images_then_videos_then_audios_preserving_order_within_type",
        "status": "submitting", "request_id": None, "manifest_path": str(manifest_path),
        "timings": {"server_full_request_s": None,
                    "server_full_request_note": "not exposed by the pinned async API; inference_time_s is reported separately"},
    }
    save_json(manifest_path, manifest)
    started = time.perf_counter()
    deadline = started + timeout

    def budget() -> float:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("client deadline exceeded; the server job may still be running")
        return min(remaining, http_timeout)

    try:
        job = read_json(server, "/v1/videos", payload=payload, timeout=budget())
        request_id = job.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise ClientError("submission did not return a job id; do not automatically resubmit")
        manifest["request_id"] = request_id
        manifest["timings"]["client_submit_s"] = time.perf_counter() - started
        path = "/v1/videos/" + urllib.parse.quote(request_id, safe="")
        print(f"SUBMITTED id={request_id} manifest={manifest_path}", flush=True)
        while True:
            if job.get("id") != request_id:
                raise ClientError("status response belongs to a different job id")
            manifest["server_job"] = job
            manifest["status"] = str(job.get("status", "unknown"))
            save_json(manifest_path, manifest)
            state = job.get("status")
            if state == "completed":
                break
            if state == "failed":
                raise ClientError(f"server generation failed: {json.dumps(job.get('error'), ensure_ascii=False)}")
            if state not in {"queued", "in_progress"}:
                raise ClientError(f"unexpected job state: {state!r}")
            time.sleep(min(poll_interval, budget()))
            job = read_json(server, path, timeout=budget())
        manifest["timings"]["client_to_completed_s"] = time.perf_counter() - started
        manifest["timings"]["server_reported_inference_s"] = job.get("inference_time_s")
        manifest["timings"]["server_stage_durations"] = job.get("stage_durations", {})
        manifest["peak_memory_mb"] = job.get("peak_memory_mb")
        manifest["served_model"] = job.get("model")
        manifest["status"] = "downloading"
        save_json(manifest_path, manifest)
        target = directory / "output.mp4"
        temporary = directory / "output.mp4.partial"
        with urllib.request.urlopen(server + path + "/content", timeout=budget()) as response:
            with temporary.open("xb") as output:
                while True:
                    budget()
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
        os.replace(temporary, target)
        # Full transport delivery, excluding local ffprobe audit time.
        manifest["timings"]["client_to_media_s"] = time.perf_counter() - started
        manifest["output"] = str(target)
        manifest["status"] = "validating"
        save_json(manifest_path, manifest)
        audit_start = time.perf_counter()
        manifest["media_validation"] = validate_media(target, payload, ffprobe=ffprobe)
        manifest["timings"]["local_validation_s"] = time.perf_counter() - audit_start
        manifest["timings"]["client_to_validated_s"] = time.perf_counter() - started
        manifest["status"] = "completed"
        save_json(manifest_path, manifest)
        print(f"DONE client_to_media={manifest['timings']['client_to_media_s']:.3f}s output={target}", flush=True)
        return manifest
    except (Exception, KeyboardInterrupt) as error:
        previous = manifest["status"]
        rejected = (previous == "submitting" and manifest["request_id"] is None
                    and isinstance(error, HTTPFailure) and 400 <= error.status < 500)
        manifest["status"] = "failed" if manifest["request_id"] or rejected else "submission_unknown"
        manifest["failed_phase"] = previous
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["timings"]["client_elapsed_s"] = time.perf_counter() - started
        manifest["server_may_still_be_running"] = (not rejected and previous in {"submitting", "queued", "in_progress"})
        save_json(manifest_path, manifest)
        raise ClientError(f"{error}; manifest={manifest_path}; no automatic resubmission or server cancellation",
                          manifest=manifest) from error


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8108")
    p.add_argument("--model", default=None, help="advertised model alias; omitted means server default")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--prompt-file", type=Path)
    for kind in ORDER:
        p.add_argument("--" + kind, action=ReferenceAction, dest="references", const=kind,
                       metavar="PATH", help="repeatable; pass images before videos before audio")
    p.add_argument("--server-paths", action="store_true",
                   help="reference paths are already on the server, not checked or hashed by this client")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--output-root", type=Path, default=Path("outputs/omni"))
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--http-timeout", type=float, default=60)
    p.add_argument("--poll-interval", type=float, default=0.25)
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--cache-condition", choices=("cold", "warm-new", "warm-fixed-ref", "warm-full-conditioning", "unspecified"), default="unspecified")
    p.add_argument("--dry-run", action="store_true", help="print JSON request; no HTTP or output writes")
    return p


def main() -> int:
    p = parser()
    a = p.parse_args()
    try:
        server = normalize_server(a.server)
        prompt = a.prompt_file.read_text() if a.prompt_file else a.prompt
        payload, audit = build_request(prompt, a.references or [], seed=a.seed, frames=a.frames,
                                       width=a.width, height=a.height, model=a.model, server_paths=a.server_paths)
        if a.dry_run:
            print(json.dumps({"server": server, "request": payload, "references": audit}, ensure_ascii=False, indent=2))
            return 0
        if not shutil.which(a.ffprobe):
            raise ClientError(f"ffprobe not found: {a.ffprobe}")
        run_job(server, payload, audit, output_root=a.output_root, timeout=a.timeout,
                poll_interval=a.poll_interval, http_timeout=a.http_timeout, ffprobe=a.ffprobe,
                cache_condition=a.cache_condition)
        return 0
    except (ClientError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
