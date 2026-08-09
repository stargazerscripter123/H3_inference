#!/usr/bin/env python
"""FL2VA via SGLang's OpenAI-compatible async video API (verified from the
official MiniMax-H3 cookbook): POST /v1/videos -> poll -> GET /content.

NFE semantics (pinned to sglang 407a65d3 / vllm-omni a874b8e09): the engine's
num_inference_steps=N produces N sigma POINTS (linspace) -> N-1 DiT forwards.
--nfe requests an exact forward count and maps to num_inference_steps = NFE+1.
If upstream changes to N-points->N-forwards semantics, update NFE_STEP_OFFSET.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

NFE_STEP_OFFSET = 1  # num_inference_steps = nfe + NFE_STEP_OFFSET (sigma-points semantics)


class VramSampler(threading.Thread):
    def __init__(self, gpus="0,1,2,3"):
        super().__init__(daemon=True)
        self.gpus = gpus.split(",")
        self.peak = {g: 0 for g in self.gpus}
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            for g in self.gpus:
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "-i", g, "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"], text=True, timeout=10)
                    self.peak[g] = max(self.peak[g], int(out.strip()))
                except Exception:
                    pass
            time.sleep(2)

    def report(self):
        return " ".join(f"gpu{g}={v}MiB" for g, v in self.peak.items())


def api(base, path, payload=None, raw=False):
    req = urllib.request.Request(base + path)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read() if raw else json.loads(r.read())


def schedule_sha256(steps: int, video_shift: float, audio_shift: float) -> str:
    """Audit fingerprint of the analytic sigma grids (author formula, float64,
    9 decimals) — both engines verified to reproduce it (test_h3_schedule.py)."""
    def grid(shift):
        n = steps - 1
        return [shift * u / (1.0 + (shift - 1.0) * u)
                for u in (1.0 - i / n for i in range(n + 1))]
    payload = ",".join(f"{s:.9f}" for s in grid(video_shift) + grid(audio_shift))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def served_model(server: str) -> str | None:
    """向服务端问它实际加载的权重路径(审计用: base 还是 merged turbo)。"""
    import urllib.request
    try:
        with urllib.request.urlopen(server.rstrip("/") + "/v1/models", timeout=10) as r:
            data = json.loads(r.read()).get("data") or []
        return data[0].get("id") if data else None
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:30010")
    p.add_argument("--first", required=True, help="absolute path on this host")
    p.add_argument("--last", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=12,
                   help="raw num_inference_steps (N sigma points -> N-1 forwards)")
    p.add_argument("--nfe", type=int, default=None,
                   help="exact DiT forward count; overrides --steps via steps=nfe+1")
    p.add_argument("--flow-shift", type=float, default=12.0)
    p.add_argument("--audio-flow-shift", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--short-edge", type=int, default=768,
                   help="cookbook-validated value is 768; other values experimental")
    p.add_argument("--quality", default=None, help="lossless|high (optional)")
    p.add_argument("--out", required=True, help="output mp4 path")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--dry-run", action="store_true",
                   help="print the JSON body and exit (payload audit)")
    a = p.parse_args()
    prompt = Path(a.prompt_file).read_text() if a.prompt_file else a.prompt
    if not prompt:
        p.error("need --prompt or --prompt-file")

    steps = (a.nfe + NFE_STEP_OFFSET) if a.nfe is not None else a.steps
    body = {
        "model": "MiniMaxAI/MiniMax-H3",
        "prompt": prompt,
        "seconds": int(round(a.seconds)),
        "task": "fl2va",
        "conditions": [
            {"type": "image", "uri": f"file://{Path(a.first).resolve()}",
             "role": "keyframe", "frame_index": 0},
            {"type": "image", "uri": f"file://{Path(a.last).resolve()}",
             "role": "keyframe", "frame_index": -1},
        ],
        "target": {"short_edge": a.short_edge, "aspect_ratio": "auto",
                   "duration_seconds": a.seconds},
        "num_outputs_per_prompt": 1,
        "num_inference_steps": steps,
        "flow_shift": a.flow_shift,
        "audio_flow_shift": a.audio_flow_shift,
        "seed": a.seed,
    }
    if a.quality:
        body["quality"] = a.quality
    if a.dry_run:
        print(json.dumps(body, ensure_ascii=False))
        return 0

    sampler = VramSampler()
    sampler.start()
    t0 = time.time()
    try:
        resp = api(a.server, "/v1/videos", body)
    except urllib.error.HTTPError as e:
        print(f"SUBMIT_FAILED HTTP_{e.code}:", e.read()[:800].decode(errors="replace"))
        return 1
    vid = resp.get("id")
    if not vid:
        print("SUBMIT_FAILED:", json.dumps(resp)[:1500])
        return 1
    print("video_id:", vid, flush=True)

    last_note = 0.0
    while True:
        time.sleep(3)
        st = api(a.server, f"/v1/videos/{vid}")
        status = st.get("status")
        if status == "completed":
            break
        if status == "failed":
            print("GENERATION_FAILED:", json.dumps(st)[:2000])
            return 2
        if time.time() - last_note > 60:
            print(f"  ... {time.time()-t0:.0f}s status={status} vram={sampler.report()}",
                  flush=True)
            last_note = time.time()
        if time.time() - t0 > a.timeout:
            print("TIMEOUT")
            return 3

    data = api(a.server, f"/v1/videos/{vid}/content", raw=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(data)
    wall = time.time() - t0
    sampler.stop_flag = True
    manifest = {
        "engine": "sglang",
        "requested_nfe": a.nfe,
        "engine_num_inference_steps": steps,
        "nfe_semantics": f"num_inference_steps = NFE + {NFE_STEP_OFFSET} (sigma-points)",
        "observed_dit_forwards": None,  # filled by bench harness from server log
        "video_flow_shift": a.flow_shift,
        "audio_flow_shift": a.audio_flow_shift,
        "sigma_schedule_sha256": schedule_sha256(steps, a.flow_shift, a.audio_flow_shift),
        "seed": a.seed, "seconds": a.seconds,
        "short_edge": a.short_edge,
        "served_model": served_model(a.server),
        "server": a.server, "wall_s": round(wall, 1),
        "peak_vram": sampler.peak,
        "out": a.out, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(a.out + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE wall={wall:.1f}s peak_vram[{sampler.report()}] outputs=['{a.out}']")
    return 0


if __name__ == "__main__":
    sys.exit(main())
