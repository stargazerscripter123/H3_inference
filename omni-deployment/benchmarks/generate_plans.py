#!/usr/bin/env python3
"""Generate a 32-measured-request Omni screening matrix; never run media or HTTP.

Only imports the existing standard-library client/benchmark for schema checks.
Does not inspect media files, connect to the server, or execute emitted commands.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import shlex
import sys


SOURCE_URL = "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov"
SEEDS = (1234, 5678)
ANCHORS = {"w": 240.0, "a": 55.0, "b": 90.0}
SCENARIOS = {
    "S1-single-image": (0.0, ("image",)),
    "S2-multiple-images": (1.0, ("image", "image")),
    "S3-image-audio": (2.0, ("image", "audio")),
    "S4-image-video": (3.0, ("image", "video")),
}
BASE_PROMPTS = {
    "S1-single-image": "Use the visual appearance and setting in <Picture 1>. Create one continuous cinematic shot with natural movement.",
    "S2-multiple-images": "Use the visual appearances and setting in <Picture 1> and <Picture 2>. Create one continuous cinematic shot with natural movement.",
    "S3-image-audio": "Use the visual appearance and setting in <Picture 1>, and the acoustic character and rhythm in <Audio 1>. Create one continuous cinematic shot with natural movement and synchronized sound.",
    "S4-image-video": "Use the visual appearance in <Picture 1> and the movement and camera style in <Video 1>. Create one continuous cinematic shot with natural movement.",
}
PROMPT_ENDINGS = {
    "w": "Keep the camera movement gentle throughout this preparatory take.",
    "new-1": "Keep the camera movement slow and steady throughout the shot.",
    "new-2": "Maintain a gentle, continuous camera move from beginning to end.",
    "fixed-1": "Use a smooth, unhurried camera move across the entire scene.",
    "fixed-2": "Let the camera drift steadily while keeping the motion natural.",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def media_manifest(input_root: str) -> tuple[list[dict], dict[tuple[str, str], list[dict]]]:
    assets = []
    references = {}
    for case_id, (offset, kinds) in SCENARIOS.items():
        for variant, anchor in ANCHORS.items():
            refs = []
            for index, kind in enumerate(kinds):
                suffix = {"image": "png", "audio": "wav", "video": "mp4"}[kind]
                stem = f"{case_id.split('-')[0].lower()}_{variant}_{kind}_{index + 1}"
                path = f"{input_root}/bench32-v1/{stem}.{suffix}"
                start = anchor + offset + (0.5 if kind == "image" and index == 1 else 0)
                asset = {
                    "id": stem, "case_id": case_id, "variant": variant, "type": kind,
                    "path": path, "source_start_seconds": start,
                    "duration_seconds": None if kind == "image" else 4.0,
                    "sha256": None, "verification_status": "not_created_or_verified",
                }
                if kind == "image":
                    asset["expected_metadata"] = {"video_streams": 1, "audio_streams": 0, "height": 480, "decoded_frames": 1}
                elif kind == "audio":
                    asset["expected_metadata"] = {"video_streams": 0, "audio_streams": 1, "sample_rate": 32000, "channels": 2, "duration_seconds": 4.0}
                else:
                    asset["expected_metadata"] = {"video_streams": 1, "audio_streams": 0, "height": 480, "fps": 24, "decoded_frames": 96, "duration_seconds": 4.0}
                assets.append(asset)
                refs.append({"type": kind, "source": path})
            references[case_id, variant] = refs
    return assets, references


def build_plan(references: dict) -> dict:
    rows = []
    for case_id in SCENARIOS:
        schedule = (
            ("w", "w", "warmup", "unspecified", 9000),
            ("new-1", "a", "measured", "warm-new", SEEDS[0]),
            ("new-2", "b", "measured", "warm-new", SEEDS[1]),
            ("fixed-1", "b", "measured", "warm-fixed-ref", SEEDS[0]),
            ("fixed-2", "b", "measured", "warm-fixed-ref", SEEDS[1]),
        )
        for label, variant, phase, condition, seed in schedule:
            rows.append({
                "name": f"{case_id}-{label}", "case_id": case_id, "phase": phase,
                "cache_condition": condition,
                "prompt": BASE_PROMPTS[case_id] + " " + PROMPT_ENDINGS[label],
                "seed": seed, "frames": 362, "width": 832, "height": 480,
                "references": references[case_id, variant],
            })
    return {"requests": rows}


def derive_commands(assets: list[dict]) -> str:
    lines = [
        "#!/usr/bin/env bash", "set -euo pipefail",
        '# Run manually on the target host; argument is the full film path or URL.',
        '# Timeline is relative to the full film, not to an already-trimmed clip.',
        ': "${1:?Usage: bash derive_refs.sh FULL_FILM_PATH_OR_URL}"',
        'source_video="$1"',
        'ffmpeg_bin="${OMNI_FFMPEG:-/home/isaac/workdir/h3/.tools/media/bin/ffmpeg}"',
        'ffprobe_bin="${OMNI_FFPROBE:-/home/isaac/workdir/h3/.tools/media/bin/ffprobe}"',
        "# -n prevents overwriting an existing derived asset.",
        "mkdir -p " + shlex.quote(str(Path(assets[0]["path"]).parent)),
    ]
    for asset in assets:
        lines.append("")
        lines.append(f"# {asset['id']} at full-film t={asset['source_start_seconds']:.3f}s")
        prefix = '"$ffmpeg_bin" -hide_banner -loglevel error -n -ss ' + str(asset["source_start_seconds"]) + ' -i "$source_video"'
        kind = asset["type"]
        if kind == "image":
            extra = ["-map", "0:v:0", "-frames:v", "1", "-an", "-vf", "scale=-2:480:flags=lanczos,setsar=1"]
        elif kind == "audio":
            extra = ["-map", "0:a:0", "-t", "4", "-vn", "-c:a", "pcm_s16le", "-ar", "32000", "-ac", "2"]
        else:
            extra = ["-map", "0:v:0", "-t", "4", "-an", "-vf", "scale=-2:480:flags=lanczos,setsar=1,fps=24", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        lines.append(prefix + " " + shlex.join(extra + [asset["path"]]))
        lines.append('"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json ' + shlex.quote(asset["path"]) + " > " + shlex.quote(asset["path"] + ".ffprobe.json"))
        lines.append("sha256sum " + shlex.quote(asset["path"]) + " > " + shlex.quote(asset["path"] + ".sha256"))
    return "\n".join(lines) + "\n"


def validate_plan(plan: dict, assets: list[dict], scripts: Path) -> dict:
    sys.path.insert(0, str(scripts))
    client = importlib.import_module("run_omni")
    benchmark = importlib.import_module("bench_omni")
    asset_paths = {asset["path"] for asset in assets}
    measured = Counter()
    prompts = set()
    seen_paths = set()
    checked = 0
    allowed_fields = {"name", "case_id", "phase", "cache_condition", "prompt", "seed", "frames", "width", "height", "references"}
    for row in plan["requests"]:
        if set(row) != allowed_fields or row["phase"] not in {"warmup", "measured"}:
            raise ValueError("Unexpected benchmark row schema")
        if row["cache_condition"] not in benchmark.CACHE_CONDITIONS:
            raise ValueError("Unsupported cache condition")
        if row["prompt"] in prompts:
            raise ValueError("Prompt reused within one persistent-service run")
        prompts.add(row["prompt"])
        paths = {ref["source"] for ref in row["references"]}
        if not paths <= asset_paths:
            raise ValueError("Reference missing from derivative manifest")
        if row["cache_condition"] == "warm-new" and paths & seen_paths:
            raise ValueError("warm-new reference path was already used")
        seen_paths.update(paths)
        payload, _ = client.build_request(
            row["prompt"], [(ref["type"], ref["source"]) for ref in row["references"]],
            seed=row["seed"], frames=row["frames"], width=row["width"], height=row["height"],
            model=None, server_paths=True,
        )
        if payload["num_inference_steps"] != 8:
            raise ValueError("Unexpected NFE contract")
        if row["phase"] == "measured":
            measured[row["case_id"], row["cache_condition"]] += 1
        checked += 1
    for case_id in SCENARIOS:
        case_rows = [row for row in plan["requests"] if row["case_id"] == case_id]
        if case_rows[2]["references"] != case_rows[3]["references"] or case_rows[3]["references"] != case_rows[4]["references"]:
            raise ValueError("Fixed-reference rows must reuse immediately primed B references")
        for condition in ("warm-new", "warm-fixed-ref"):
            subset = [row for row in case_rows if row["cache_condition"] == condition]
            if sorted(row["seed"] for row in subset) != sorted(SEEDS):
                raise ValueError("Each case/cache cell must have both seeds")
    if len(measured) != 8 or set(measured.values()) != {2}:
        raise ValueError("Wrong screening matrix size")
    return {
        "status": "schema_dry_validation_passed", "rows_per_profile": checked,
        "measured_per_profile": sum(measured.values()), "warmups_per_profile": checked - sum(measured.values()),
        "profiles": ["sp1", "sp2"], "total_measured": 32, "total_explicit_warmups": 8,
        "derivative_assets": len(assets), "api_payloads_built_without_network": checked,
        "run_omni_sha256": hashlib.sha256((scripts / "run_omni.py").read_bytes()).hexdigest(),
        "bench_omni_sha256": hashlib.sha256((scripts / "bench_omni.py").read_bytes()).hexdigest(),
        "media_exists_or_hash_verified": False, "media_opened": False,
        "gpu_generation_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--scripts-dir", type=Path, default=Path(__file__).resolve().parent.parent / "scripts")
    parser.add_argument("--input-root", default="/home/isaac/workdir/h3/inputs/tos")
    args = parser.parse_args()
    if not args.input_root.startswith("/") or "\x00" in args.input_root:
        parser.error("--input-root must be an absolute server directory")
    assets, references = media_manifest(args.input_root.rstrip("/"))
    plan = build_plan(references)
    evidence = validate_plan(plan, assets, args.scripts_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for profile in ("sp1", "sp2"):
        write_json(args.output_dir / f"screen32-{profile}.json", plan)
    write_json(args.output_dir / "derivatives.json", {
        "source_url": SOURCE_URL, "license": "CC BY 3.0",
        "attribution": "Tears of Steel — Blender Foundation | mango.blender.org",
        "license_source": "https://mango.blender.org/about/",
        "changes": "Timestamp excerpts; resized images/video; video soundtrack removed; audio resampled to 32-kHz stereo.",
        "source_timeline": "full-film seconds", "source_file_sha256": None,
        "assets": assets,
        "must_verify_before_run": [
            "Full-film source provenance and hash, or URL plus source excerpt evidence",
            "Every derivative SHA-256 and ffprobe metadata match its declared type/geometry/duration",
            "All same-type derivatives have distinct SHA-256; different paths alone do not establish new content",
            "No benchmark reference content was used earlier in this service process",
        ],
    })
    (args.output_dir / "derive_refs.sh").write_text(derive_commands(assets), encoding="utf-8")
    write_json(args.output_dir / "dry-validation.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
