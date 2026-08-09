#!/usr/bin/env python
"""把一个 MiniMax-H3 FL2VA 任务提交给本机的 headless ComfyUI 并等结果。

图的形状照官方 video_minimax_h3_i2v 模板:BasicGuider + SamplerCustomAdvanced,
res_multistep / simple,**没有 CFG**(单分支,这是延迟只有一半的前提)。

一般不直接调它,走 scripts/h3.sh。直接调的话最少要给 --te --first --last
--prompt-file --prefix。

步数语义
    --nfe = 实际 DiT forward 次数。**在 ComfyUI 侧 steps == forwards**,所以
    nfe 直接就是 sampler 的 steps,不加一。
    注意这个等式只在 ComfyUI 成立:vLLM-Omni / SGLang 的 num_inference_steps=N
    只跑 N-1 次 forward(N 个 sigma 点),那边的客户端要 +1。照抄会差一步。

输出
    stdout 最后一行是 `DONE wall=…s peak_vram[…] outputs=[…]`,这是给调用方
    解析的唯一数据通道(h3.sh 靠它拿产物路径)。**不要改这行的格式**;另外
    输出文件名里不能有逗号,解析是按逗号切的。
    同时写一份 <prefix>.manifest.json 到 outputs/ 下,记录这次跑的全部参数与
    环境指纹 —— 本仓库没有 README,manifest 就是唯一的运行记录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# ComfyUI 侧 steps == forwards。留成常量是为了让"为什么这里不 +1"有处可查。
NFE_STEP_OFFSET = 0

# 两个 VAE 都是必需的:H3 是视频 + 原生音频联合生成,少一个图就断了。
# audio VAE 必须是 fp32 那份 —— 换 fp16 会静默出错(不报错,音频变噪声)。
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def build_graph(a: argparse.Namespace) -> dict:
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": a.dit, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": a.te, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
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
        # ★ Turbo LoRA 必须走作者的 MiniMaxH3TurboLoRA 节点。
        #
        # 千万不要换成 ComfyUI 核心的 LoraLoaderModelOnly:核心的
        # comfy/lora.py::model_lora_keys_unet 只认 `diffusion_model.<k>` 与
        # `lora_unet_<k>` 两种前缀,没有 MiniMaxH3 分支。这个 LoRA 的键是裸键,
        # 一个都匹配不上 —— 结果是加载 0/518 个张量**而且完全不报错**,
        # 你会得到一个"看起来在跑但其实没加 LoRA"的 4 步结果(糊成一片)。
        #
        # 另外 pruned/量化基座还需要节点在运行时注入 51 个 adaln,核心节点更不会做。
        # 验收标志:worker 日志里应出现
        #   [MiniMaxH3TurboLoRA] pruned base [bypass]: ... 208 backbone modules,
        #   158 bypass adapters, 1 injections, 50 int8 fc2 via merge + 51 adaln injected
        # 没有这一行就是没生效。install/doctor.sh 把它当断言。
        g["18"] = {"class_type": "MiniMaxH3TurboLoRA",
                   "inputs": {"model": ["1", 0],
                              "lora_name": a.turbo_lora,
                              "strength": a.lora_strength,
                              "low_vram": a.lora_low_vram}}
        # 配套少步采样器。它会自适应 ComfyUI 版本:本部署钉的 ComfyUI 还没有原生
        # ModelSamplingAV,所以会回落到节点自带的 legacy dual-schedule
        # (video shift 12 / audio shift 3),日志里能看到 sigmas=[1.0, 0.973, ...]。
        # 升级 ComfyUI 会切到原生实现,**输出会变** —— 见 install/pins.env。
        g["9"] = {"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
        g["8"]["inputs"]["model"] = ["18", 0]
        g["11"]["inputs"]["model"] = ["18", 0]
    if a.teacache_thresh > 0:
        # TeaCache 是有损的跳步缓存。total_steps 必须等于 sampler 的 steps,
        # 否则跳步窗口算错。
        #
        # jitter 不是玄学:ComfyUI 会缓存节点输出,输入完全相同的第二次提交会直接
        # 复用上一次的节点实例,导致 TeaCache 内部状态不重置、加速**静默失效**
        # (不报错,只是变慢或结果不对)。给阈值加一个纳秒级抖动即可让节点重建。
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
    """按物理 GPU 序号采样显存峰值(逗号分隔,可多张)。"""

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
    p = argparse.ArgumentParser(
        description="提交 H3 FL2VA 图给本机 ComfyUI 并等结果(一般由 scripts/h3.sh 调用)")
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--dit", default="minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    p.add_argument("--te", required=True, help="ComfyUI/models/text_encoders/ 下的文件名")
    p.add_argument("--first", required=True, help="ComfyUI input 目录里的文件名")
    p.add_argument("--last", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None, help="从文件读 prompt(优先于 --prompt)")
    p.add_argument("--width", type=int, default=864)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--length", type=int, default=124, help="帧数;124 帧 @24fps = 5.17s")
    p.add_argument("--nfe", type=int, default=None,
                   help="实际 DiT forward 次数。ComfyUI 侧 steps==forwards,所以直接当 steps 用")
    p.add_argument("--steps", type=int, default=30, help="与 --nfe 等价;两者都给以 --nfe 为准")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="h3/h3", help="SaveVideo 的 filename_prefix")
    p.add_argument("--gpu-index", default="0", help="采样显存用的物理 GPU 序号(逗号分隔)")
    p.add_argument("--turbo-lora", default=None,
                   help="ComfyUI/models/loras/ 下的 Turbo LoRA 文件名;配合 --nfe 4~8")
    p.add_argument("--lora-low-vram", action="store_true",
                   help="把 LoRA merge 进权重省显存(量化基座上结果更软);默认 bypass 运行时应用,最锐")
    p.add_argument("--lora-strength", type=float, default=1.0, help="作者调校值 1.0")
    p.add_argument("--teacache-thresh", type=float, default=0.0,
                   help="TeaCache rel_l1_thresh;0 = 关闭")
    p.add_argument("--teacache-start", type=int, default=2)
    p.add_argument("--teacache-end", type=int, default=-2)
    p.add_argument("--manifest-dir", default=None,
                   help="manifest 落盘目录;不给则不写")
    p.add_argument("--profile", default=None, help="仅记进 manifest,便于事后追溯")
    p.add_argument("--timeout", type=int, default=7200)
    a = p.parse_args()

    if a.nfe is not None:
        a.steps = a.nfe + NFE_STEP_OFFSET
    if a.prompt_file:
        a.prompt = Path(a.prompt_file).read_text()
    if not a.prompt:
        p.error("需要 --prompt 或 --prompt-file")

    stats = api(a.server, "/system_stats")
    comfy_ver = stats["system"]["comfyui_version"]
    devices = [d["name"] for d in stats["devices"]]
    print("server up:", comfy_ver, "| devs:", devices)

    graph = build_graph(a)
    graph_sha = hashlib.sha256(
        json.dumps(graph, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    sampler = VramSampler(a.gpu_index)
    sampler.start()
    t0 = time.time()
    resp = api(a.server, "/prompt", {"prompt": graph, "client_id": "h3_local"})
    pid = resp.get("prompt_id")
    if not pid:
        print("SUBMIT_FAILED:", json.dumps(resp)[:2000])
        return 1
    print("prompt_id:", pid, "| graph_sha:", graph_sha)

    last_note = 0.0
    while True:
        time.sleep(5)
        hist = api(a.server, f"/history/{pid}")
        if pid in hist:
            status = hist[pid].get("status", {})
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

    if a.manifest_dir:
        # 没有 README,manifest 就是这次运行的唯一记录。记到"事后能重现"的粒度:
        # 参数 + 图指纹 + 环境指纹(ComfyUI 版本/设备)+ 结果。
        man = {
            "profile": a.profile,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "comfyui_version": comfy_ver,
            "devices": devices,
            "graph_sha16": graph_sha,
            "prompt_id": pid,
            "params": {
                "nfe": a.nfe, "steps": a.steps, "seed": a.seed,
                "width": a.width, "height": a.height, "length": a.length,
                "dit": a.dit, "te": a.te, "video_vae": VIDEO_VAE, "audio_vae": AUDIO_VAE,
                "turbo_lora": a.turbo_lora, "lora_strength": a.lora_strength,
                "lora_low_vram": a.lora_low_vram,
                "teacache_thresh": a.teacache_thresh or None,
                "first": a.first, "last": a.last,
                "prompt_sha16": hashlib.sha256(a.prompt.encode()).hexdigest()[:16],
            },
            # ComfyUI 侧 steps==forwards;引擎侧是 NFE+1。写进 manifest 免得
            # 事后拿这份记录和引擎侧的数字直接比。
            "nfe_semantics": "comfyui: steps == forwards (NFE_STEP_OFFSET=0)",
            "wall_s": round(wall, 2),
            "peak_vram": sampler.peak,
            "outputs": outputs,
        }
        d = Path(a.manifest_dir)
        d.mkdir(parents=True, exist_ok=True)
        mp = d / (os.path.basename(a.prefix) + ".manifest.json")
        mp.write_text(json.dumps(man, indent=2, ensure_ascii=False))
        print("manifest:", mp)

    # ★ 调用方解析这一行。格式别动;输出文件名里不能有逗号。
    print(f"DONE wall={wall:.1f}s peak_vram[{sampler.report()}] outputs={outputs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
