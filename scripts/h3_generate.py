#!/usr/bin/env python3
"""MiniMax H3 FL2VA — 从 Mac Studio 一键提交首帧+尾帧+prompt 到 GPU 机器生成视频.

用法示例:
  # 双机并行(默认),自动回传到 outputs/<name>/
  scripts/h3_generate.py --first a.png --last b.png --prompt "..." --name mytest

  # 只跑 5090,prompt 从文件读,8 秒,portrait 画布自动按输入方向选
  scripts/h3_generate.py --first a.png --last b.png --prompt-file p.txt \
      --host 5090 --seconds 8 --seed 42

流程: 本地 ffmpeg 统一画布(cover+center-crop,规避 H3 首帧 stretch/尾帧 crop
的不对称) -> scp 上传 -> 远端生成(ComfyUI 图执行 或 vLLM/SGLang serving,直播进度)
-> 回传 MP4。结果段分别报告启动(切换+就绪)与推理耗时。

哪台机器支持哪个 profile: `scripts/h3_generate.py --list`(矩阵由 HOSTS 配置派生)。

依赖: 本机 ffmpeg / ssh / scp;远端已按本项目部署(~/data/dropbox/CV/h3)。
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT / "outputs"

# 每台机器的固定配置(与已部署的 launch/runner 脚本一致)
HOSTS = {
    "5090": {
        "ssh": "popos-5090",
        "server": "http://127.0.0.1:8188",
        "launch": "bash ~/data/dropbox/CV/h3/scripts/launch_comfy.sh 1 8188",
        "te": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "gpu_index": "1",
        "switch_script": "h3_switch_5090.sh",
        "switch_baseline": "comfy",
        # 5090 TP2 serving(2026-08-09 修复): PR5910 的 resident-repoint 丢 stride bug
        # 已定位并打补丁(scripts/pr5910_resident_stride_fix.patch),TP2+FP8+DLO
        # warm 46.9s、画质经帧对比验证;详见 doc/speedup_5090_serving_results.md round-2 章。
        "serving": {
            "vllm-fp8-original-tp2": {
                "switch": "vllm", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 11},
            # Turbo-LoRA merged + TP2 FP8+DLO(2026-08-09): NFE6 29.0s / NFE4 22.0s
            "vllm-fp8-turbo-tp2": {
                "switch": "vllm-turbo", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 6},
        },
        "comfy": {
            # 零 tweak 参照(默认 SDPA attention),用 host 默认 server/launch/gpu_index/te
            "comfy-int8-original-1c": {"steps": 30},
            # 独立 worker(GPU0:8189)+ 低步数 + TeaCache(有损缓存)
            "comfy-int8-teacache-1c": {
                "server": "http://127.0.0.1:8189",
                "launch": "bash ~/data/dropbox/CV/h3/scripts/launch_comfy_5090_turbo.sh",
                "gpu_index": "0",
                "extra_args": " --teacache-thresh 0.10",
                "steps": 12,
                "canvas": (864, 480),   # 竖版自动转 480x864
            },
            # Turbo LoRA 原生路线(作者的 ComfyUI 节点),独立 worker GPU0:8190。
            # ComfyUI 步数语义 steps==forwards,所以 nfe 直接就是 steps(不 +1)。
            "comfy-int8-turbo-1c": {
                "server": "http://127.0.0.1:8190",
                "switch": "comfy-tlora",   # 独占启动: host RAM 装不下三个 worker
                "launch": "bash ~/data/dropbox/CV/h3/scripts/launch_comfy_5090_tlora.sh",
                "gpu_index": "0",
                "extra_args": " --turbo-lora minimax_h3_turbo_v4_step600_ema.safetensors",
                "nfe": 6,
                "canvas": (864, 480),
            },
        },
    },
    "6000a": {
        "ssh": "popos-6000a",
        "server": "http://127.0.0.1:8288",
        "launch": "bash ~/data/dropbox/CV/h3/scripts/launch_comfy_6000a_4gpu.sh",
        "te": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "gpu_index": "0,1,2,3",
        "switch_script": "h3_switch.sh",
        "switch_baseline": "baseline",
        # 非 ComfyUI serving 后端(互斥占卡, h3_switch.sh 管理)
        # 步数语义用 NFE(实际 DiT forward 数,客户端换算 num_inference_steps=NFE+1)
        "serving": {
            "vllm-bf16-original-tp4": {
                "switch": "vllm bf16", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 11},
            "vllm-fp8-original-tp4": {
                "switch": "vllm fp8", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 11},
            "vllm-fp8-turbo-tp4": {
                "switch": "vllm-turbo fp8", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 6},
            "vllm-bf16-turbo-tp4": {
                "switch": "vllm-turbo bf16", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 6},
            # SGLang API 硬锁短边 768,画布参数不生效(实际 1376×768)
            "sglang-fp8-original-tp4": {
                "switch": "sglang", "client": "run_fl2va_sglang.py",
                "health": "http://127.0.0.1:30010/health", "nfe": 11},
            "sglang-fp8-turbo-tp4": {
                "switch": "sglang-turbo", "client": "run_fl2va_sglang.py",
                "health": "http://127.0.0.1:30010/health", "nfe": 6},
        },
        "comfy": {
            "comfy-int8-original-1c": {"steps": 30},
            # 全 BF16 零 tweak 诊断参照: DisTorch2 四卡分置
            "comfy-bf16-original-4c": {
                "steps": 30,
                "model_args": (
                    " --dit minimax_h3_fl2va_bf16.safetensors"
                    " --te qwen3vl_32b_minimax_h3_bf16.safetensors"
                    " --dit-compute cuda:2 --dit-vvram 35 --dit-donor cuda:3"
                    " --te-compute cuda:0 --te-vvram 30 --te-donor cuda:1"
                    " --vae-device cuda:1"
                ),
            },
        },
    },
    # RunPods 4×5090 云机(2026-08-09, doc/TP4_5090.md;无 ComfyUI,仅 serving)。
    # 布局在 /workspace/h3(持久卷);pod 重建用 scripts/setup_runpods.sh。
    "runpods": {
        "ssh": "5090-Runpods",
        "remote_base": "/workspace/h3",
        "remote_python": "/workspace/h3/env/bin/python",
        "switch_script": "h3_switch_runpods.sh",
        # 该机的 switch 还收第 4 个位置参数 [resident](DLO 常驻层数),
        # 不给则按精度取默认(bf16=40, fp8=50)。它只影响显存/调度、不改变输出,
        # 因此按命名规范**不进 profile 名**,由 h3_eval.py --resident 扫描。
        "resident_arg": "positional",
        # h3_switch_runpods.sh 的三个维度正交: <original|turbo> <bf16|fp8> <tp4|tp2u2>
        # 该机无 GPU P2P(topo 全 CNS),NCCL 走主机中转,所以 TP4 不必然最快 —— tp2u2
        # (TP2+USP2,配对落在同一 NUMA)是有意义的对照,实测反而更快。
        #
        # 显存: switch 脚本头注释说的 "BF16 TP2 = 33G 装不下" 指的是**全常驻**;
        # 实际开了 DLO 只驻留 resident 层,BF16 TP2×U2 @ r40 实测峰值 28.4G/卡
        # (32.6G 里余 ~4G) —— **装得下**,故四种组合全部暴露。
        # 注意 bf16+tp2u2 的余量只有 ~4G,把 resident 往上调(如 r50)有 OOM 风险;
        # 不显式给 --resident 时用 switch 默认(bf16=40 / fp8=50),那是验证过的档位。
        "serving": {
            "vllm-bf16-original-tp4": {
                "switch": "original bf16 tp4", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 11},
            "vllm-fp8-original-tp4": {
                "switch": "original fp8 tp4", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 11},
            "vllm-fp8-original-tp2u2": {
                "switch": "original fp8 tp2u2", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 11},
            "vllm-bf16-original-tp2u2": {
                "switch": "original bf16 tp2u2", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 11},
            "vllm-bf16-turbo-tp4": {
                "switch": "turbo bf16 tp4", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 6},
            "vllm-fp8-turbo-tp4": {
                "switch": "turbo fp8 tp4", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 6},
            "vllm-fp8-turbo-tp2u2": {
                "switch": "turbo fp8 tp2u2", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 6},
            # 实测 r40: NFE6 20.6s / 峰值 28.4G 卡(装得下)
            "vllm-bf16-turbo-tp2u2": {
                "switch": "turbo bf16 tp2u2", "client": "run_fl2va_vllm.py",
                "health": "http://127.0.0.1:8091/health",
                "canvas": (864, 480), "nfe": 6},
        },
    },
}
REMOTE_BASE = "data/dropbox/CV/h3"  # 相对远端 $HOME

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


# profile 命名: <引擎>-<精度>-<变体…>-<拓扑>
#   精度   = DiT 权重精度 int8|fp8|bf16|nvfp4(TE 精度不同时记文档,不进名字)
#   变体   = 会改变输出或步数的技术选择,可叠加,按 <加速方法…>-turbo 排序:
#            original(零 tweak) / teacache|sage|flash(显式加速方法) / turbo(Turbo LoRA)
#            纯内存调度 flag(async-offload 等)不进名字,它们不改变输出。
#            vLLM/SGLang 的 attention backend 属引擎必选配置,也不进名字。
#   拓扑   = 1c/4c(ComfyUI 计算卡数) | tp2/tp4(张量并行度)

# 旧名 -> 新名。**必须按机器区分**: 旧的 "turbo" 在 5090 指 ComfyUI TeaCache、
# 在 6000a 指 vLLM FP8 —— 正是这套命名要消除的歧义,这里把它显式化。
LEGACY_ALIASES = {
    "5090": {
        "baseline": "comfy-int8-original-1c",
        "turbo": "comfy-int8-teacache-1c",
        "vllm": "vllm-fp8-original-tp2",
        "turbo-lora": "vllm-fp8-turbo-tp2",
        "vllm-turbo": "vllm-fp8-turbo-tp2",
    },
    "6000a": {
        "baseline": "comfy-int8-original-1c",
        "bf16": "comfy-bf16-original-4c",
        "turbo": "vllm-fp8-original-tp4",
        "vllm": "vllm-bf16-original-tp4",
        "sglang": "sglang-fp8-original-tp4",
        "turbo-lora": "vllm-fp8-turbo-tp4",
        "vllm-turbo": "vllm-fp8-turbo-tp4",
        "sglang-turbo": "sglang-fp8-turbo-tp4",
    },
    "runpods": {
        "turbo-lora": "vllm-bf16-original-tp4",
    },
}


def host_profiles(cfg: dict) -> dict:
    """该机器支持的全部 profile -> 定义(serving 与 comfy 合并)。"""
    out = {p: {**d, "_kind": "serving"} for p, d in cfg.get("serving", {}).items()}
    out.update({p: {**d, "_kind": "comfy"} for p, d in cfg.get("comfy", {}).items()})
    return out


def all_profiles() -> list[str]:
    seen: dict[str, None] = {}
    for cfg in HOSTS.values():
        for p in host_profiles(cfg):
            seen[p] = None
    return sorted(seen)


def _subseq(query: list[str], full: list[str]) -> bool:
    """query 的各段按序出现在 full 中(家族简称: 允许省略中间/末尾字段)。"""
    it = iter(full)
    return all(any(q == f for f in it) for q in query)


def resolve_profile(host: str, name: str) -> tuple[str | None, str]:
    """(canonical, 说明)。不支持返回 (None, 原因)。"""
    avail = host_profiles(HOSTS[host])
    if name in avail:
        return name, ""
    alias = LEGACY_ALIASES.get(host, {}).get(name)
    if alias and alias in avail:
        return alias, f"旧名 {name} → {alias}"
    hits = [p for p in avail if _subseq(name.split("-"), p.split("-"))]
    if len(hits) == 1:
        return hits[0], f"家族简称 {name} → {hits[0]}"
    if len(hits) > 1:
        return None, f"{name} 在 {host} 上有多个候选,请写全名: {', '.join(sorted(hits))}"
    return None, f"{host} 不支持 {name}"


def profile_backend(cfg: dict, profile: str) -> str | None:
    """该 profile 在该机器上走哪条后端;不支持返回 None。"""
    d = host_profiles(cfg).get(profile)
    if d is None:
        return None
    if d["_kind"] == "serving":
        return f"serving:{d['switch']}"
    if "model_args" in d:
        return "comfy:DisTorch 4卡"
    port = (d.get("server") or cfg["server"]).rsplit(":", 1)[-1]
    return f"comfy:gpu{d.get('gpu_index', cfg.get('gpu_index'))}:{port}"


def print_capabilities() -> None:
    """打印 host × profile 能力矩阵(从 HOSTS 派生,不会与实现脱节)。"""
    hosts = list(HOSTS)
    profs = all_profiles()
    w = max(len(p) for p in profs) + 2
    print("host × profile 能力矩阵(✓=该机器原生支持)\n")
    print(" " * w + "".join(f"{h:>10}" for h in hosts))
    for p in profs:
        cells = [f"{'✓' if p in host_profiles(HOSTS[h]) else '·':>10}" for h in hosts]
        print(f"{p:<{w}}" + "".join(cells))
    print("\n后端与默认步数:")
    for h in hosts:
        print(f"  {h} ({HOSTS[h]['ssh']}):")
        for p, d in sorted(host_profiles(HOSTS[h]).items()):
            nfe, steps = d.get("nfe"), d.get("steps")
            dflt = f"NFE {nfe}" if nfe else (f"{steps} 步" if steps else "-")
            if d.get("client") == "run_fl2va_sglang.py":
                canvas = "1376×768(API 锁 768 短边)"
            else:
                cw, chh = d.get("canvas", (832, 480))
                canvas = f"{cw}×{chh}"
            print(f"    {p:<26} → {profile_backend(HOSTS[h], p):<24} {dflt:<7} {canvas}")
    print("\n旧名仍可用(按机器解析):")
    for h, m in LEGACY_ALIASES.items():
        print(f"  {h}: " + ", ".join(f"{o}→{n}" for o, n in sorted(m.items())))
    print("\n家族简称: 省略字段按 token 子序列匹配,如 vllm-fp8-turbo 在 5090 得 -tp2、"
          "6000a 得 -tp4;有歧义时报错并列出候选。")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def probe_size(img: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(img)],
        text=True).strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def snap_length(seconds: float) -> int:
    """帧数必须满足 n % 17 == 5(24fps);训练范围约 124-362。"""
    n = round(seconds * 24)
    n = n + ((5 - n % 17) % 17)
    if n < 124:
        print(f"[warn] {seconds}s 低于训练下限,提升到 124 帧 (~5.17s)")
        n = 124
    if n > 362:
        print(f"[warn] {n} 帧超出训练范围上限 362 (~15s),质量未验证")
    return n


def preprocess(src: Path, dst: Path, w: int, h: int) -> None:
    """scale-to-cover + center-crop 到目标画布,两帧同一变换,无形变。"""
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    sh(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
        "-frames:v", "1", "-vf", vf, str(dst)])


def run_serving(tag: str, cfg: dict, sv: dict, prof: str, job: str, workdir: Path,
                a: argparse.Namespace, results: dict) -> None:
    """vLLM/SGLang serving 后端: 切换(互斥)-> 等健康 -> 远端客户端 -> 回传."""
    ssh_host = cfg["ssh"]
    prefix = f"[{tag}:{prof}]"
    # 远端根与 python 可按 host 覆盖(RunPods 等非 dropbox 布局的机器)
    rb = cfg.get("remote_base", REMOTE_BASE)
    rpy = cfg.get("remote_python", "~/miniconda3/envs/h3_comfy_NV_py312/bin/python")
    try:
        sh(["scp", "-q", *SSH_OPTS,
            str(workdir / "first.png"), str(workdir / "last.png"),
            f"{ssh_host}:{rb}/inputs/"])
        sh(["scp", "-q", *SSH_OPTS, str(workdir / "prompt.txt"),
            f"{ssh_host}:{rb}/workflows/{job}_prompt.txt"])
        switch_script = cfg.get("switch_script", "h3_switch.sh")
        print(f"{prefix} 切换后端 ({sv['switch']}),冷启动可能需要几分钟...", flush=True)
        t_start = time.time()  # 启动计时: 切换命令 + 健康就绪
        sh(["ssh", *SSH_OPTS, ssh_host,
            f"bash {rb}/scripts/{switch_script} {sv['switch']}"],
           stdout=subprocess.DEVNULL)
        t0 = time.time()
        while True:
            r = subprocess.run(["ssh", *SSH_OPTS, ssh_host,
                                f"curl -s -m 3 {sv['health']} >/dev/null 2>&1"],
                               capture_output=True)
            if r.returncode == 0:
                break
            if time.time() - t0 > 1200:
                results[tag] = ("FAILED", "backend health timeout (20min)")
                return
            print(f"{prefix} 等待后端就绪 {int(time.time()-t0)}s ...", flush=True)
            time.sleep(20)
        startup_s = time.time() - t_start
        out_rel = f"outputs/{job}/{job}_{tag}_{prof}.mp4"
        if sv["client"] == "run_fl2va_vllm.py":
            extra = f" --width {a.canvas_w} --height {a.canvas_h}"
        else:
            extra = " --short-edge 768"  # SGLang API 只接受 768 短边
        if a.nfe is not None:
            steps_arg = f" --nfe {a.nfe}"
        elif sv.get("nfe") is not None and not getattr(a, "steps_explicit", False):
            steps_arg = f" --nfe {sv['nfe']}"
        else:
            steps_arg = f" --steps {a.steps}"
        cmd = (f"{rpy}"
               f" {rb}/scripts/{sv['client']}"
               f" --first {rb}/inputs/first.png"
               f" --last {rb}/inputs/last.png"
               f" --prompt-file {rb}/workflows/{job}_prompt.txt"
               f" --seconds {a.seconds}{steps_arg} --seed {a.seed}{extra}"
               f" --out {rb}/{out_rel}")
        t_inf = time.time()  # 推理计时: 远端客户端全程(优先取其上报的 wall)
        proc = subprocess.Popen(["ssh", *SSH_OPTS, ssh_host, cmd],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        ok = False
        wall_s = None
        peak_vram = None
        for line in proc.stdout:
            line = line.rstrip()
            print(f"{prefix} {line}", flush=True)
            if line.startswith("DONE"):
                ok = True
                m = re.search(r"wall=([0-9.]+)s", line)
                if m:
                    wall_s = float(m.group(1))
                mv = re.search(r"peak_vram\[([^\]]*)\]", line)
                if mv:
                    peak_vram = mv.group(1)
        proc.wait()
        infer_s = wall_s if wall_s is not None else round(time.time() - t_inf, 1)
        if not ok or proc.returncode != 0:
            results[tag] = ("FAILED", f"client exit={proc.returncode}")
            return
        local = workdir / f"{job}_{tag}_{prof}.mp4"
        sh(["scp", "-q", *SSH_OPTS, f"{ssh_host}:{rb}/{out_rel}", str(local)])
        results[tag] = ("OK", str(local.relative_to(PROJECT)),
                        {"startup_s": round(startup_s), "infer_s": infer_s,
                         "peak_vram": peak_vram})
    except subprocess.CalledProcessError as e:
        results[tag] = ("FAILED", str(e))


def run_host(tag: str, cfg: dict, job: str, workdir: Path, a: argparse.Namespace,
             results: dict) -> None:
    ssh_host = cfg["ssh"]
    prefix = f"[{tag}]"

    def ssh_run(remote_cmd: str, **kw):
        return sh(["ssh", *SSH_OPTS, ssh_host, remote_cmd], **kw)

    prof, note = resolve_profile(tag, a.profile)
    if prof is None:
        results[tag] = ("FAILED", note)
        return
    if note:
        print(f"{prefix} [note] {note}")
    pdef = host_profiles(cfg)[prof]
    prefix = f"[{tag}:{prof}]"
    if pdef["_kind"] == "serving":
        run_serving(tag, cfg, pdef, prof, job, workdir, a, results)
        return

    try:
        # 0) 互斥: 回 ComfyUI 产线前先停 serving 后端(模型冷重载属预期)
        t_start = time.time()  # 启动计时: 停 serving + ComfyUI 拉起
        if "serving" in cfg:
            ss = cfg.get("switch_script", "h3_switch.sh")
            # comfy profile 可声明自己的 switch 模式(如 comfy-tlora 只起 :8190 那一个
            # worker) —— host RAM 装不下多个 worker 共存
            sb = pdef.get("switch", cfg.get("switch_baseline", "baseline"))
            ssh_run(f"bash ~/{REMOTE_BASE}/scripts/{ss} {sb}",
                    stdout=subprocess.DEVNULL)
        # 1) 确保该 profile 的 ComfyUI worker 在线(幂等)
        ssh_run(pdef.get("launch", cfg["launch"]), stdout=subprocess.DEVNULL)
        startup_s = time.time() - t_start

        # 2) 上传素材与 prompt
        sh(["scp", "-q", *SSH_OPTS,
            str(workdir / "first.png"), str(workdir / "last.png"),
            f"{ssh_host}:{REMOTE_BASE}/inputs/"])
        sh(["scp", "-q", *SSH_OPTS, str(workdir / "prompt.txt"),
            f"{ssh_host}:{REMOTE_BASE}/workflows/{job}_prompt.txt"])

        # 3) 前台运行 runner,直播输出(profile 定义可覆盖 server/gpu_index/模型参数)
        server = pdef.get("server", cfg["server"])
        gpu_index = pdef.get("gpu_index", cfg["gpu_index"])
        model_args = pdef.get("model_args", f" --te {cfg['te']}") + pdef.get("extra_args", "")
        # ComfyUI 步数语义 steps==forwards(不像引擎侧要 +1),所以 nfe 直接当 steps 用
        steps = a.nfe if a.nfe is not None else (
            a.steps if a.steps_explicit else (pdef.get("steps") or pdef.get("nfe") or 30))
        runner = (
            f"cd ~/{REMOTE_BASE} && "
            f"~/miniconda3/envs/h3_comfy_NV_py312/bin/python scripts/run_fl2va.py"
            f" --server {server}"
            f"{model_args}"
            f" --first first.png --last last.png"
            f" --prompt-file workflows/{job}_prompt.txt"
            f" --width {a.canvas_w} --height {a.canvas_h} --length {a.length}"
            f" --steps {steps} --seed {a.seed}"
            f" --gpu-index {gpu_index}"
            f" --prefix {job}/{job}_{prof}"
        )
        t_inf = time.time()  # 推理计时: 远端 runner 全程(含图执行/VAE/封装)
        proc = subprocess.Popen(["ssh", *SSH_OPTS, ssh_host, runner],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        outputs: list[str] = []
        wall_s = None
        peak_vram = None
        for line in proc.stdout:
            line = line.rstrip()
            print(f"{prefix} {line}", flush=True)
            if line.startswith("DONE") and "outputs=[" in line:
                seg = line.split("outputs=[", 1)[1].rstrip("]")
                outputs = [s.strip(" '\"") for s in seg.split(",") if s.strip(" '\"")]
                m = re.search(r"wall=([0-9.]+)s", line)
                if m:
                    wall_s = float(m.group(1))
                mv = re.search(r"peak_vram\[([^\]]*)\]", line)
                if mv:
                    peak_vram = mv.group(1)
        proc.wait()
        infer_s = wall_s if wall_s is not None else round(time.time() - t_inf, 1)
        if proc.returncode != 0 or not outputs:
            results[tag] = ("FAILED", f"runner exit={proc.returncode}")
            return

        # 4) 回传结果
        local_files = []
        for rel in outputs:
            local = workdir / f"{job}_{tag}_{prof}{Path(rel).suffix or '.mp4'}"
            sh(["scp", "-q", *SSH_OPTS,
                f"{ssh_host}:{REMOTE_BASE}/outputs/{rel}", str(local)])
            local_files.append(local)
        results[tag] = ("OK", ", ".join(str(f.relative_to(PROJECT)) for f in local_files),
                        {"startup_s": round(startup_s), "infer_s": infer_s,
                         "peak_vram": peak_vram})
    except subprocess.CalledProcessError as e:
        results[tag] = ("FAILED", str(e))


def main() -> int:
    # --list 先于必填参数校验处理(纯查询,不需要素材)
    if "--list" in sys.argv[1:]:
        print_capabilities()
        return 0

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="打印 host × profile 能力矩阵后退出(不需要 --first/--last)")
    ap.add_argument("--first", required=True, type=Path, help="首帧图片(本机路径)")
    ap.add_argument("--last", required=True, type=Path, help="尾帧图片(本机路径)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="prompt 文本")
    g.add_argument("--prompt-file", type=Path, help="prompt 文本文件")
    ap.add_argument("--host", choices=["5090", "6000a", "runpods", "both"], default="both",
                    help="both=5090+6000a;runpods 需显式指定(云机,仅 serving profile)")
    ap.add_argument("--name", default=None, help="任务名(默认时间戳)")
    ap.add_argument("--seconds", type=float, default=5.0, help="时长秒数(默认 5)")
    ap.add_argument("--steps", type=int, default=None,
                    help="denoise 步数(默认见 profile;--list 可查)")
    ap.add_argument("--nfe", type=int, default=None,
                    help="精确 DiT forward 次数。引擎侧(vLLM/SGLang)客户端换算 "
                         "num_inference_steps=NFE+1;ComfyUI 侧 steps==forwards 直接用。"
                         "Turbo 产线推荐 4-8,默认 6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default=None, metavar="PROFILE",
                    help="命名: <引擎>-<精度>-<变体…>-<拓扑>,如 comfy-int8-turbo-1c / "
                         "vllm-fp8-turbo-tp4。支持家族简称(省略字段,如 vllm-fp8-turbo "
                         "在两台机器上各自解析)与旧名(baseline/turbo/vllm/sglang/bf16/"
                         "turbo-lora/sglang-turbo,按机器映射)。全量见 --list。"
                         "serving 产线互斥占卡,切换含分钟级冷启动")
    ap.add_argument("--quality", choices=["int8", "bf16"], default=None,
                    help=argparse.SUPPRESS)  # 兼容别名: int8->baseline, bf16->bf16
    ap.add_argument("--width", type=int, default=None,
                    help="画布宽(默认按 profile 定义 + 输入方向)")
    ap.add_argument("--height", type=int, default=None)
    a = ap.parse_args()
    if a.profile is None:
        a.profile = "bf16" if a.quality == "bf16" else "baseline"

    for tool in ("ffmpeg", "ffprobe", "ssh", "scp"):
        if not shutil.which(tool):
            print(f"缺少 {tool}", file=sys.stderr)
            return 1
    for f in (a.first, a.last):
        if not f.exists():
            print(f"文件不存在: {f}", file=sys.stderr)
            return 1
    a.steps_explicit = a.steps is not None

    # profile 逐机解析(同一简称在不同机器可解析成不同全名)
    hosts = ["5090", "6000a"] if a.host == "both" else [a.host]
    resolved = {}
    for t in hosts:
        prof, note = resolve_profile(t, a.profile)
        resolved[t] = prof
        if prof is None:
            print(f"[note] {t}: {note}")
    if not any(resolved.values()):
        print(f"没有机器支持 --profile {a.profile};用 --list 看能力矩阵", file=sys.stderr)
        return 1

    # 画布:显式指定 > profile 定义(各机须一致) > 832×480
    if a.width and a.height:
        cw, ch = a.width, a.height
        if cw % 32 or ch % 32:
            print("宽高必须是 32 的倍数", file=sys.stderr)
            return 1
    else:
        wants = {host_profiles(HOSTS[t]).get(p, {}).get("canvas", (832, 480))
                 for t, p in resolved.items() if p}
        if len(wants) > 1:
            print(f"各机 profile 的默认画布不一致 {sorted(wants)},请显式指定 "
                  f"--width/--height", file=sys.stderr)
            return 1
        base_w, base_h = wants.pop()
        w, h = probe_size(a.first)
        cw, ch = (base_w, base_h) if w >= h else (base_h, base_w)
    a.canvas_w, a.canvas_h = cw, ch
    a.length = snap_length(a.seconds)

    job = a.name or time.strftime("h3_%m%d_%H%M%S")
    job = "".join(c if c.isalnum() or c in "_-" else "_" for c in job)
    workdir = OUTPUTS / job
    workdir.mkdir(parents=True, exist_ok=True)

    if a.nfe is not None:
        steps_desc = f"nfe {a.nfe}"
    elif a.steps_explicit:
        steps_desc = f"steps {a.steps}"
    else:
        per = []
        for t, p in resolved.items():
            if not p:
                continue
            d = host_profiles(HOSTS[t])[p]
            per.append(f"{t}:{'nfe ' + str(d['nfe']) if d.get('nfe') else str(d.get('steps')) + ' 步'}")
        steps_desc = "profile 默认(" + ", ".join(per) + ")"
    plan = ", ".join(f"{t}→{p or '不支持'}" for t, p in resolved.items())
    print(f"任务 {job}: 画布 {cw}x{ch}, {a.length} 帧 (~{a.length/24:.2f}s), "
          f"{steps_desc}, seed {a.seed}")
    print(f"  profile 解析: {plan}")
    preprocess(a.first, workdir / "first.png", cw, ch)
    preprocess(a.last, workdir / "last.png", cw, ch)
    prompt_text = a.prompt_file.read_text() if a.prompt_file else a.prompt
    (workdir / "prompt.txt").write_text(prompt_text)

    hosts = [t for t in hosts if resolved[t]]   # 不支持的机器已在上面提示,直接跳过
    results: dict = {}
    threads = [threading.Thread(target=run_host,
                                args=(t, HOSTS[t], job, workdir, a, results))
               for t in hosts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n===== 结果 =====")
    ok = True
    for tag in hosts:
        status, detail, *extra = results.get(tag, ("FAILED", "no result"))
        mark = "✅" if status == "OK" else "❌"
        timing = ""
        if extra and isinstance(extra[0], dict):
            t = extra[0]
            timing = (f"  ⏱ 启动 {t['startup_s']}s(切换+就绪) "
                      f"+ 推理 {t['infer_s']}s = 共 {round(t['startup_s'] + t['infer_s'], 1)}s")
        print(f"{mark} {tag}: {detail}{timing}")
        ok = ok and status == "OK"
    if ok:
        print(f"\n输出目录: {workdir}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
