#!/usr/bin/env python
"""Sanity-check a MiniMax-H3 Turbo LoRA file (larryvrh/MiniMax-H3-Turbo-Lora).

Gate G0: exact key set (259 pairs), all BF16, shapes match the H3 FL2VA
architecture, alpha convention documented. Read-only; exits nonzero on any
mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys

from safetensors import safe_open

HIDDEN = 5376
INNER = 56 * 128          # 7168 attention inner dim
FFN = 14336
TIME_EMBED = 2688
ADALN_OUT = 96768
FINAL_ADALN_OUT = 10752


def expected_targets():
    """{module_name: (A_in, B_out)} for every expected LoRA pair."""
    t = {}
    for i in range(50):
        t[f"blocks.{i}.attn.qkv_proj"] = (HIDDEN, 3 * INNER)
        t[f"blocks.{i}.attn.out_proj"] = (INNER, HIDDEN)
        t[f"blocks.{i}.mlp.fc1"] = (HIDDEN, 2 * FFN)
        t[f"blocks.{i}.mlp.fc2"] = (FFN, HIDDEN)
        t[f"blocks.{i}.adaln_proj.linear"] = (TIME_EMBED, ADALN_OUT)
    for i in range(2):
        t[f"token_refiner.blocks.{i}.attn.qkv_proj"] = (HIDDEN, 3 * INNER)
        t[f"token_refiner.blocks.{i}.attn.out_proj"] = (INNER, HIDDEN)
        t[f"token_refiner.blocks.{i}.mlp.fc1"] = (HIDDEN, 2 * FFN)
        t[f"token_refiner.blocks.{i}.mlp.fc2"] = (FFN, HIDDEN)
    t["final_layer.adaln_proj.linear"] = (TIME_EMBED, FINAL_ADALN_OUT)
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("lora")
    args = p.parse_args()
    exp = expected_targets()
    errors = []
    ranks = {}
    with safe_open(args.lora, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        meta = f.metadata() or {}
        print(f"metadata: {json.dumps(meta, ensure_ascii=False)}")
        print(f"total tensors: {len(keys)}")

        alpha_keys = [k for k in keys if k.endswith(".alpha")]
        pair_bases = {k[: -len(".lora_A.weight")] for k in keys if k.endswith(".lora_A.weight")}
        if len(pair_bases) != len(exp):
            errors.append(f"pair count {len(pair_bases)} != expected {len(exp)}")
        if pair_bases != set(exp):
            errors.append(f"key set mismatch; unexpected={sorted(pair_bases - set(exp))[:5]} "
                          f"missing={sorted(set(exp) - pair_bases)[:5]}")
        if len(keys) != 2 * len(pair_bases) + len(alpha_keys):
            errors.append(f"stray tensors: {len(keys)} != 2*{len(pair_bases)}+{len(alpha_keys)}")

        for base in sorted(pair_bases & set(exp)):
            in_dim, out_dim = exp[base]
            sa = f.get_slice(base + ".lora_A.weight")
            sb = f.get_slice(base + ".lora_B.weight")
            a_shape, b_shape = sa.get_shape(), sb.get_shape()
            if sa.get_dtype() != "BF16" or sb.get_dtype() != "BF16":
                errors.append(f"{base}: dtype {sa.get_dtype()}/{sb.get_dtype()} != BF16")
            r = a_shape[0]
            ranks[base] = r
            if a_shape != [r, in_dim]:
                errors.append(f"{base}: lora_A {a_shape} != [{r},{in_dim}]")
            if b_shape != [out_dim, r]:
                errors.append(f"{base}: lora_B {b_shape} != [{out_dim},{r}]")

    rank_hist = {}
    for base, r in ranks.items():
        kind = "adaln" if "adaln" in base else "linear"
        rank_hist.setdefault((kind, r), 0)
        rank_hist[(kind, r)] += 1
    print("rank histogram:", {f"{k}:r{r}": c for (k, r), c in sorted(rank_hist.items())})
    if alpha_keys:
        print(f"alpha tensors present: {len(alpha_keys)}")
    else:
        print("alpha metadata ABSENT -> per author/Comfy convention alpha=rank (scale 1.0)")

    if errors:
        print(f"\nFAILED ({len(errors)} errors):")
        for e in errors[:20]:
            print("  -", e)
        return 1
    print(f"\nOK: {len(pair_bases)} pairs, all BF16, all shapes match H3 FL2VA architecture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
