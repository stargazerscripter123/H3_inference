#!/usr/bin/env python
"""Submit a MiniMax H3 FL2VA job to a headless ComfyUI and wait for the result.

Builds the API-format graph (matching the official video_minimax_h3_i2v template:
BasicGuider + SamplerCustomAdvanced, res_multistep/simple, no CFG), submits it,
polls until completion, reports wall time + peak VRAM + output files.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request

def build_graph(a: argparse.Namespace) -> dict:
    # Component-to-device placement (requires ComfyUI-MultiGPU custom node):
    # any of --dit-device/--te-device/--vae-device switches that loader to its
    # MultiGPU variant with an explicit device input.
    if a.dit_compute:
        # DisTorch2: weights sharded to donor via P2P, compute on one device
        unet = {"class_type": "UNETLoaderDisTorch2MultiGPU",
                "inputs": {"unet_name": a.dit, "weight_dtype": "default",
                           "compute_device": a.dit_compute,
                           "virtual_vram_gb": a.dit_vvram,
                           "donor_device": a.dit_donor}}
    elif a.dit_device:
        unet = {"class_type": "UNETLoaderMultiGPU",
                "inputs": {"unet_name": a.dit, "weight_dtype": "default", "device": a.dit_device}}
    else:
        unet = {"class_type": "UNETLoader",
                "inputs": {"unet_name": a.dit, "weight_dtype": "default"}}
    if a.te_compute:
        clip = {"class_type": "CLIPLoaderDisTorch2MultiGPU",
                "inputs": {"clip_name": a.te, "type": "minimax",
                           "device": a.te_compute,
                           "virtual_vram_gb": a.te_vvram,
                           "donor_device": a.te_donor}}
    elif a.te_device:
        clip = {"class_type": "CLIPLoaderMultiGPU",
                "inputs": {"clip_name": a.te, "type": "minimax", "device": a.te_device}}
    else:
        clip = {"class_type": "CLIPLoader",
                "inputs": {"clip_name": a.te, "type": "minimax", "device": "default"}}
    def vae_loader(name):
        if a.vae_device:
            return {"class_type": "VAELoaderMultiGPU",
                    "inputs": {"vae_name": name, "device": a.vae_device}}
        return {"class_type": "VAELoader", "inputs": {"vae_name": name}}
    g = {
        "1": unet,
        "2": clip,
        "3": vae_loader("minimax_h3_video_vae_fp16.safetensors"),
        "4": vae_loader("minimax_h3_audio_vae_fp32.safetensors"),
        "5": {"class_type": "LoadImage", "inputs": {"image": a.first}},
        "6": {"class_type": "LoadImage", "inputs": {"image": a.last}},
        "7": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": a.prompt,
                         "width": a.width, "height": a.height, "length": a.length,
                         "first_frame": ["5", 0], "last_frame": ["6", 0]}},
        "8": {"class_type": "BasicScheduler",
              "inputs": {"model": ["1", 0], "scheduler": "simple",
                         "steps": a.steps, "denoise": 1.0}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
        "11": {"class_type": "BasicGuider",
               "inputs": {"model": ["1", 0], "conditioning": ["7", 0]}},
        "12": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["10", 0], "guider": ["11", 0], "sampler": ["9", 0],
                          "sigmas": ["8", 0], "latent_image": ["7", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "15": {"class_type": "CreateVideo",
               "inputs": {"images": ["13", 0], "fps": 24.0, "audio": ["14", 0]}},
        "16": {"class_type": "SaveVideo",
               "inputs": {"video": ["15", 0], "filename_prefix": a.prefix,
                          "format": "auto", "codec": "auto"}},
    }
    if a.turbo_lora:
        # Turbo LoRA 少步蒸馏: 插在 UNETLoader 与所有 MODEL 消费者之间
        g["18"] = {"class_type": "MiniMaxH3TurboLoRA",
                   "inputs": {"model": ["1", 0],
                              "lora_name": a.turbo_lora,
                              "strength": a.lora_strength,
                              "low_vram": a.lora_low_vram}}
        # 配套的少步采样器(节点会自适应 ComfyUI 版本处理音频 schedule)
        g["9"] = {"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
        g["8"]["inputs"]["model"] = ["18", 0]
        g["11"]["inputs"]["model"] = ["18", 0]
    if a.teacache_thresh > 0:
        # TeaCache (lossy step-skip cache). total_steps must equal sampler steps.
        # Tiny per-run jitter on the threshold busts ComfyUI's node-output cache —
        # otherwise the node's state is not reset and caching silently disables
        # on repeat submissions with identical inputs.
        jitter = (time.time_ns() % 997) * 1e-9
        g["17"] = {"class_type": "MiniMaxH3TeaCache",
                   "inputs": {"model": ["18", 0] if a.turbo_lora else ["1", 0],
                              "rel_l1_thresh": round(a.teacache_thresh + jitter, 12),
                              "start_step": a.teacache_start,
                              "end_step": a.teacache_end,
                              "total_steps": a.steps}}
        g["8"]["inputs"]["model"] = ["17", 0]
        g["11"]["inputs"]["model"] = ["17", 0]
    return g

class VramSampler(threading.Thread):
    """Tracks per-GPU peak VRAM for a comma-separated list of physical indices."""

    def __init__(self, gpu_indices: str):
        super().__init__(daemon=True)
        self.gpus = [g.strip() for g in gpu_indices.split(",") if g.strip()]
        self.peak = {g: 0 for g in self.gpus}
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            for g in self.gpus:
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "-i", g,
                         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        text=True, timeout=10)
                    self.peak[g] = max(self.peak[g], int(out.strip()))
                except Exception:
                    pass
            time.sleep(2)

    def report(self) -> str:
        return " ".join(f"gpu{g}={v}MiB" for g, v in self.peak.items())

def api(base: str, path: str, payload: dict | None = None):
    req = urllib.request.Request(base + path)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--dit", default="minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    p.add_argument("--te", required=True)
    p.add_argument("--first", required=True, help="filename inside ComfyUI input dir")
    p.add_argument("--last", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None, help="read prompt text from file (wins over --prompt)")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--length", type=int, default=124)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="h3_smoke/h3_smoke")
    p.add_argument("--gpu-index", default="0", help="comma-separated physical GPU indices for VRAM sampling")
    p.add_argument("--dit-device", default=None)
    p.add_argument("--dit-compute", default=None, help="DisTorch2 compute device for DiT")
    p.add_argument("--dit-vvram", type=float, default=0.0, help="GB of DiT weights to park on donor")
    p.add_argument("--dit-donor", default="cpu")
    p.add_argument("--te-compute", default=None, help="DisTorch2 compute device for TE")
    p.add_argument("--te-vvram", type=float, default=0.0)
    p.add_argument("--te-donor", default="cpu")
    p.add_argument("--turbo-lora", default=None,
                   help="ComfyUI/models/loras/ 下的 Turbo LoRA 文件名;"
                        "配合 --steps 4-8 使用(ComfyUI 语义 steps==forwards)")
    p.add_argument("--lora-low-vram", action="store_true",
                   help="把 LoRA merge 进权重以省显存(量化基座上更软);"
                        "默认 bypass 运行时应用,最锐")
    p.add_argument("--lora-strength", type=float, default=1.0,
                   help="Turbo LoRA 强度,作者调校值为 1.0")
    p.add_argument("--teacache-thresh", type=float, default=0.0,
                   help="TeaCache rel_l1_thresh; 0 disables (lossless baseline)")
    p.add_argument("--teacache-start", type=int, default=2)
    p.add_argument("--teacache-end", type=int, default=-2)
    p.add_argument("--te-device", default=None)
    p.add_argument("--vae-device", default=None)
    p.add_argument("--timeout", type=int, default=7200)
    a = p.parse_args()
    if a.prompt_file:
        from pathlib import Path
        a.prompt = Path(a.prompt_file).read_text()
    if not a.prompt:
        p.error("need --prompt or --prompt-file")

    stats = api(a.server, "/system_stats")
    print("server up:", stats["system"]["comfyui_version"],
          "| devs:", [d["name"] for d in stats["devices"]])

    graph = build_graph(a)
    sampler = VramSampler(a.gpu_index)
    sampler.start()
    t0 = time.time()
    resp = api(a.server, "/prompt", {"prompt": graph, "client_id": "h3_smoke"})
    pid = resp.get("prompt_id")
    if not pid:
        print("SUBMIT_FAILED:", json.dumps(resp)[:2000])
        return 1
    print("prompt_id:", pid)

    last_note = 0.0
    while True:
        time.sleep(5)
        hist = api(a.server, f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("completed"):
                break
            if status.get("status_str") == "error":
                print("EXECUTION_ERROR")
                for m in status.get("messages", []):
                    if m[0] == "execution_error":
                        print(json.dumps(m[1], indent=2)[:4000])
                return 2
        if time.time() - last_note > 60:
            q = api(a.server, "/prompt")
            print(f"  ... running {time.time()-t0:.0f}s, queue={q.get('exec_info')}, "
                  f"peak_vram={sampler.report()}", flush=True)
            last_note = time.time()
        if time.time() - t0 > a.timeout:
            print("TIMEOUT")
            return 3

    wall = time.time() - t0
    sampler.stop_flag = True
    outputs = []
    for node_out in hist[pid].get("outputs", {}).values():
        for key in ("images", "video", "videos"):
            for f in node_out.get(key, []):
                outputs.append(f.get("subfolder", "") + "/" + f["filename"])
    print(f"DONE wall={wall:.1f}s peak_vram[{sampler.report()}] outputs={outputs}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
