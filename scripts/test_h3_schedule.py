#!/usr/bin/env python
"""Three-way sigma-schedule consistency test (Gate G0).

Compares, for NFE 4/6/8 and shifts video=12 / audio=3:
  1. the author's formula (larryvrh generate.py: shift_sigma(1 - i/n, s), n = NFE)
  2. vLLM-Omni  minimax_h3_time_shift_sigmas(num_steps=NFE+1, shift_scale=s)
  3. SGLang     minimax_h3_time_shift_sigmas(num_steps=NFE+1, shift_scale=s)

Both engines use sigma-POINTS semantics: num_inference_steps=N -> linspace(1,0,N)
-> N-1 DiT forwards. Hence engine steps = NFE + 1. Requires: identical length
(NFE+1) and max_abs_diff <= 1e-7 against the author grid.

Engine modules are loaded directly from their source files (they only import
torch), so this runs in any env with torch — no engine package import needed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import glob
import os
import sys

# 引擎源码根。各机布局不同(popos 用 $H3_ROOT/src/,runpods 用 /workspace/h3/src-*/),
# 所以按 H3_SRC_ROOT 环境变量定位,并对常见布局做回退搜索 —— 写死绝对路径会让这个
# 测试在别的机器上变成永远跑不起来的死代码。
_H3_ROOT = os.path.expanduser(os.environ.get("H3_ROOT", "~/data/dropbox/CV/h3"))
# 搜索根:显式指定 > 项目根下的 src/ 与 src-*(runpods 用 src-vllm-omni 这种平铺布局)
_SEARCH_ROOTS = [os.environ["H3_SRC_ROOT"]] if os.environ.get("H3_SRC_ROOT") else [
    os.path.join(_H3_ROOT, "src"), _H3_ROOT]

VLLM_GLOBS = [os.path.join(r, "*vllm-omni*", "vllm_omni", "diffusion", "models",
                           "minimax_h3", "time_request.py") for r in _SEARCH_ROOTS]
SGLANG_GLOBS = [os.path.join(r, "*sglang*", "python", "sglang", "multimodal_gen",
                             "**", "minimax_h3", "time_request.py")
                for r in _SEARCH_ROOTS]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def author_grid(nfe: int, shift: float) -> list[float]:
    return [shift * u / (1.0 + (shift - 1.0) * u)
            for u in (1.0 - i / nfe for i in range(nfe + 1))]


def schedule_sha256(sigmas: list[float]) -> str:
    return hashlib.sha256(",".join(f"{s:.9f}" for s in sigmas).encode()).hexdigest()[:16]


def checkout_label(path: str, marker: str, fallback: str) -> str:
    """把 .../src/<checkout>/<marker>/... 里的 <checkout> 取出来当标签。
    一台机器上常同时存在多个 checkout(如 sglang 与 sglang-pr33681),标签用来区分。"""
    head = path.split(os.sep + marker + os.sep)[0]
    return os.path.basename(head) or fallback


def main():
    # 多个 checkout 全部参测,而不是挑一个:同机上"打了补丁的"与"没打补丁的"两份
    # 并存是常态,只测其中一份等于放过另一份。
    engines = {}
    vllm_hits = [p for g in VLLM_GLOBS for p in sorted(glob.glob(g))]
    if not vllm_hits:
        print("找不到 vllm-omni 的 time_request.py。设 H3_SRC_ROOT 指向引擎源码根"
              "(或 H3_ROOT 指向项目根);已搜索:\n  " + "\n  ".join(VLLM_GLOBS),
              file=sys.stderr)
        return 1
    for i, p in enumerate(vllm_hits):
        label = checkout_label(p, "vllm_omni", "vllm-omni")
        engines[label] = load_module(p, f"vllm_tr{i}").minimax_h3_time_shift_sigmas
    # SGLang 是可选的:不是每台机器都装(如 runpods 只有 vLLM),缺了就少测一路,不算失败
    sglang_hits = [p for g in SGLANG_GLOBS for p in sorted(glob.glob(g, recursive=True))]
    for i, p in enumerate(sglang_hits):
        label = checkout_label(p, os.path.join("python", "sglang"), "sglang")
        engines[label] = load_module(p, f"sgl_tr{i}").minimax_h3_time_shift_sigmas
    if not sglang_hits:
        print("[note] 本机未装 sglang,只比对 vllm-omni 与作者公式", file=sys.stderr)
    print(f"[note] 参测引擎 checkout: {', '.join(engines)}", file=sys.stderr)
    failures = 0
    for nfe in (4, 6, 8):
        for label, shift in (("video", 12.0), ("audio", 3.0)):
            ref = author_grid(nfe, shift)
            print(f"NFE={nfe} {label}(shift={shift}) author  n={len(ref)} "
                  f"sha={schedule_sha256(ref)} {[round(s, 6) for s in ref]}")
            for eng, fn in engines.items():
                got = fn(num_steps=nfe + 1, shift_scale=shift)
                diff = max(abs(a - b) for a, b in zip(ref, got)) if len(got) == len(ref) else None
                ok = len(got) == len(ref) and diff <= 1e-7
                print(f"  {eng:10s} n={len(got)} max_abs_diff={diff} {'OK' if ok else 'FAIL'}")
                if not ok:
                    failures += 1
    if failures:
        print(f"\nFAILED: {failures} mismatches")
        return 1
    print(f"\nOK: engine steps = NFE + 1 reproduces the author sigma grid exactly "
          f"({', '.join(engines)}; video+audio; NFE 4/6/8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
