#!/usr/bin/env python3
"""Submit an explicit sequential Omni request plan to one already-running server.

No hidden warmups, server restarts, cache clearing, or failed-request retries.
Each plan item declares phase=warmup|measured and its own prompt, references,
seed and cache condition. Supply genuinely new prompts/reference content for
warm-new, or new prompts with fixed references for warm-fixed-ref. These are
operator labels; upstream API cache hits are not observable here.

Plan JSON: {"requests": [{"name": "single-image-warmup", "case_id": "single-image", "phase": "warmup",
  "prompt": "A scene based on <Picture 1>.", "seed": 1, "frames": 124,
  "references": [{"type": "image", "source": "/server/inputs/a.png"}],
  "cache_condition": "warm-full-conditioning"}]}
Reference ordering and media validation follow run_omni.py. Summaries group by
case_id, cache condition and output geometry. Set case_id to keep distinct
scenarios separate; omitted case IDs are explicitly grouped as all-unspecified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import uuid

import run_omni

CACHE_CONDITIONS = {"cold", "warm-new", "warm-fixed-ref", "warm-full-conditioning", "unspecified"}


def summarize(results: list[dict], threshold: float) -> list[dict]:
    groups = defaultdict(list)
    for result in results:
        if result["phase"] == "measured":
            key = (result.get("case_id", "all-unspecified"), result["cache_condition"],
                   result["width"], result["height"], result["frames"])
            groups[key].append(result)
    output = []
    for key, members in sorted(groups.items()):
        times = sorted(x["client_to_media_s"] for x in members if x["ok"])
        on_time = sum(value <= threshold for value in times)
        output.append({
            "case_id": key[0], "cache_condition": key[1], "width": key[2], "height": key[3], "frames": key[4],
            "requests": len(members), "valid": len(times), "failed": len(members) - len(times),
            "valid_on_time": on_time, "valid_on_time_rate": on_time / len(members),
            "threshold_s": threshold,
            "p50_s_successes_only": statistics.median(times) if times else None,
            "p95_s_successes_only": times[math.ceil(0.95 * len(times)) - 1] if times else None,
            "max_s_successes_only": max(times) if times else None,
            "timing_scope": "client submission through complete media download; local validation excluded",
        })
    return output


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--server", default="http://127.0.0.1:8108")
    p.add_argument("--model")
    p.add_argument("--server-paths", action="store_true")
    p.add_argument("--output-root", type=Path, default=Path("outputs/omni-bench"))
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--threshold", type=float, default=15.0)
    p.add_argument("--poll-interval", type=float, default=0.25)
    a = p.parse_args()
    try:
        if not shutil.which("ffprobe"):
            raise ValueError("ffprobe is required before starting a benchmark")
        raw = json.loads(a.plan.read_text())
        requests = raw["requests"]
        if not isinstance(requests, list) or not requests:
            raise ValueError("plan.requests must be a non-empty list")
        # Validate the whole plan before submitting any job.
        ready = []
        for item in requests:
            if not isinstance(item, dict):
                raise ValueError("every request must be a JSON object")
            if item.get("phase") not in {"warmup", "measured"}:
                raise ValueError("every request must specify phase=warmup or measured")
            case_id = item.get("case_id", "all-unspecified")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError("case_id must be a nonempty string when provided")
            condition = item.get("cache_condition", "unspecified")
            if condition not in CACHE_CONDITIONS:
                raise ValueError(f"invalid cache condition: {condition}")
            payload, audit = run_omni.build_request(
                item["prompt"], [(ref["type"], ref["source"]) for ref in item["references"]],
                seed=item["seed"], frames=item.get("frames", 362),
                width=item.get("width", 832), height=item.get("height", 480),
                model=a.model, server_paths=a.server_paths)
            ready.append((item, case_id, condition, payload, audit))
        server = run_omni.normalize_server(a.server)
        run_omni.require_positive_finite(timeout=a.timeout, threshold=a.threshold, poll_interval=a.poll_interval)
        root = a.output_root.resolve() / (datetime.now(timezone.utc).strftime("bench_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12])
        root.mkdir(parents=True, exist_ok=False)
        run_omni.save_json(root / "plan.json", raw)
        report = {"plan_sha256": run_omni.sha256_file(a.plan), "server": server,
                  "model": a.model, "planned_requests": len(ready), "results": [], "summary": []}
        for index, (item, case_id, condition, payload, audit) in enumerate(ready):
            result = {"index": index, "name": item.get("name", str(index)),
                      "phase": item["phase"], "case_id": case_id, "cache_condition": condition,
                      "frames": payload["num_frames"], "width": payload["width"], "height": payload["height"],
                      "seed": payload["seed"], "ok": False}
            try:
                manifest = run_omni.run_job(server, payload, audit, output_root=root,
                                           timeout=a.timeout, poll_interval=a.poll_interval,
                                           cache_condition=condition)
                result.update(ok=True, manifest_path=manifest["manifest_path"],
                              client_to_media_s=manifest["timings"]["client_to_media_s"])
            except run_omni.ClientError as error:
                result["error"] = str(error)
                print(f"FAILED {result['name']}: {error}", file=sys.stderr)
            report["results"].append(result)
            report["summary"] = summarize(report["results"], a.threshold)
            report["completed_requests"] = len(report["results"])
            report["unrun_requests"] = len(ready) - len(report["results"])
            run_omni.save_json(root / "report.json", report)
            # A failed/unknown submission can leave an active job. Default
            # fail-fast avoids queueing a second request behind unknown work.
            if not result["ok"]:
                break
        print(json.dumps({"report": str(root / "report.json"), "summary": report["summary"]}, indent=2))
        return 0 if len(report["results"]) == len(ready) and all(x["ok"] for x in report["results"]) else 1
    except (KeyError, TypeError, ValueError, OSError, run_omni.ClientError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
