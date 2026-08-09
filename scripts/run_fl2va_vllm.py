#!/usr/bin/env python
"""FL2VA via vLLM-Omni's sync video API. Uses curl for the multipart POST
(matches the officially validated request path exactly).

NFE semantics (pinned to vllm-omni a874b8e09 / sglang 407a65d3): the engine's
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


def fmt_shift(v: float) -> str:
    """Render 12.0 as '12' so default payloads stay byte-identical to the
    pre-NFE version of this script."""
    return str(int(v)) if float(v).is_integer() else str(v)


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
    p.add_argument("--server", default="http://127.0.0.1:8091")
    p.add_argument("--first", required=True)
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
    p.add_argument("--width", type=int, default=864)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--dry-run", action="store_true",
                   help="print the curl command and exit (payload audit)")
    a = p.parse_args()
    prompt = Path(a.prompt_file).read_text() if a.prompt_file else a.prompt
    if not prompt:
        p.error("need --prompt or --prompt-file")

    steps = (a.nfe + NFE_STEP_OFFSET) if a.nfe is not None else a.steps
    out = Path(a.out)
    extra = json.dumps({"task": "fl2va", "duration": a.seconds,
                        "frame_indices": [0, -1], "audio_flow_shift": a.audio_flow_shift})
    cmd = [
        "curl", "-sS", "-m", str(a.timeout), "-X", "POST",
        f"{a.server}/v1/videos/sync",
        "-F", f"prompt={prompt}",
        "-F", f"width={a.width}", "-F", f"height={a.height}", "-F", "fps=24",
        "-F", f"num_inference_steps={steps}",
        "-F", f"flow_shift={fmt_shift(a.flow_shift)}", "-F", f"seed={a.seed}",
        "-F", f"extra_params={extra}",
        "-F", f"input_references=@{a.first};type=image/png",
        "-F", f"input_references=@{a.last};type=image/png",
        "-o", str(out),
        "-w", "http=%{http_code}",
    ]
    if a.dry_run:
        print(json.dumps({"cmd": cmd}, ensure_ascii=False))
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    sampler = VramSampler()
    sampler.start()
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    sampler.stop_flag = True
    status = r.stdout.strip()
    if r.returncode != 0 or "http=200" not in status:
        body = out.read_bytes()[:800] if out.exists() else b""
        print(f"REQUEST_FAILED {status} rc={r.returncode} err={r.stderr[:300]} body={body!r}")
        return 1
    head = out.read_bytes()[:64]
    if b"ftyp" not in head:
        print(f"UNEXPECTED_RESPONSE head={head!r}")
        return 2
    manifest = {
        "engine": "vllm-omni",
        "requested_nfe": a.nfe,
        "engine_num_inference_steps": steps,
        "nfe_semantics": f"num_inference_steps = NFE + {NFE_STEP_OFFSET} (sigma-points)",
        "observed_dit_forwards": None,  # filled by bench harness from server log
        "video_flow_shift": a.flow_shift,
        "audio_flow_shift": a.audio_flow_shift,
        "sigma_schedule_sha256": schedule_sha256(steps, a.flow_shift, a.audio_flow_shift),
        "seed": a.seed, "seconds": a.seconds,
        "width": a.width, "height": a.height,
        "served_model": served_model(a.server),
        "server": a.server, "wall_s": round(wall, 1),
        "peak_vram": sampler.peak,
        "out": str(out), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE wall={wall:.1f}s peak_vram[{sampler.report()}] outputs=['{out}']")
    return 0


if __name__ == "__main__":
    sys.exit(main())
