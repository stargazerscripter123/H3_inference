#!/usr/bin/env python
"""Merge MiniMax-H3 Turbo LoRA into the official BF16 FL2VA transformer shards.

Layout facts (verified 2026-08-08, see doc/turbo_lora_results.md):
- LoRA lora_B rows are in runtime-continuous order: qkv = [Q_all|K_all|V_all],
  fc1 = [gate;up] (trained against the Comfy-Org single-file BF16 checkpoint,
  whose qkv_proj rows are continuous).
- HF disk shards store qkv per-head interleaved: <heads> groups x [q|k|v] each
  of <head_dim> rows. Both engines reorder disk->continuous at load time
  (vllm-omni minimax_h3_transformer.py:142-171; sglang _install_qkv_weight_loader).
- Therefore ONLY qkv deltas need the inverse reorder before adding to the disk
  tensor; every other target (out_proj/fc1/fc2/adaln/final_layer/token_refiner
  non-qkv) adds directly. The merged directory keeps disk layout, so one merged
  checkpoint serves both engines.

Modes:
  build (default): stream 13 shards -> <dst>.building -> verify -> manifest
                   + .complete -> atomic rename to <dst>
  --verify-only:   re-run the 3-layer verification against an existing <dst>

Verification layers (feedback: avoid self-referential checks):
  1. bit-level recompute of every modified tensor + bit-equality of unmodified
  2. independent layout oracle: Comfy single-file BF16 (the LoRA's training
     base, continuous layout) must equal reorder(HF disk tensor) bit-for-bit
  3. single-layer forward oracle: y_merged vs y_dynamic (cosine/relL2)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# qkv row-order transforms (parameterized from transformer/config.json)
# ----------------------------------------------------------------------
def reorder_grouped_to_continuous(w: torch.Tensor, groups: int, head_dim: int) -> torch.Tensor:
    """Disk grouped [g0(q|k|v), g1(q|k|v), ...] -> runtime [Q_all; K_all; V_all]."""
    per_group = 3 * head_dim
    rest = w.shape[1:]
    g = w.reshape(groups, per_group, *rest)
    q, k, v = torch.split(g, [head_dim] * 3, dim=1)
    return torch.cat([t.reshape(groups * head_dim, *rest) for t in (q, k, v)], dim=0)


def inverse_reorder_continuous_to_grouped(w: torch.Tensor, groups: int, head_dim: int) -> torch.Tensor:
    """Runtime [Q_all; K_all; V_all] -> disk grouped, exact inverse of the above."""
    rest = w.shape[1:]
    q, k, v = torch.split(w, [groups * head_dim] * 3, dim=0)
    q, k, v = (t.reshape(groups, head_dim, *rest) for t in (q, k, v))
    return torch.cat([q, k, v], dim=1).reshape(groups * 3 * head_dim, *rest)


def roundtrip_selftest(groups: int, head_dim: int) -> None:
    x = torch.arange(groups * 3 * head_dim * 7, dtype=torch.float32).reshape(-1, 7)
    assert torch.equal(reorder_grouped_to_continuous(
        inverse_reorder_continuous_to_grouped(x, groups, head_dim), groups, head_dim), x)
    assert torch.equal(inverse_reorder_continuous_to_grouped(
        reorder_grouped_to_continuous(x, groups, head_dim), groups, head_dim), x)
    log("qkv reorder round-trip self-test passed")


def is_qkv(name: str) -> bool:
    return name.endswith(".attn.qkv_proj.weight")


# ----------------------------------------------------------------------
# LoRA loading
# ----------------------------------------------------------------------
def load_lora_pairs(lora_path: str, strength: float, allow_nonunit_scale: bool):
    """Return {target_weight_name: (A fp32, B fp32, rank, alpha, scale)}."""
    sd = load_file(lora_path)
    alpha_keys = [k for k in sd if k.endswith(".alpha")]
    pairs = {}
    for k in sd:
        if not k.endswith(LORA_A_SUFFIX):
            continue
        base = k[: -len(LORA_A_SUFFIX)]
        b_key = base + LORA_B_SUFFIX
        assert b_key in sd, f"missing lora_B for {base}"
        A, B = sd[k], sd[b_key]
        assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16, base
        rank = A.shape[0]
        assert B.shape[1] == rank, f"rank mismatch {base}: A{list(A.shape)} B{list(B.shape)}"
        alpha_key = base + ".alpha"
        if alpha_key in sd:
            alpha = float(sd[alpha_key].item())
            alpha_src = "tensor"
        else:
            alpha = float(rank)  # author/Comfy convention: alpha == rank -> scale 1.0
            alpha_src = "absent->convention alpha=rank"
        scale = (alpha / rank) * strength
        if abs(scale - 1.0) > 1e-9 and not allow_nonunit_scale:
            raise SystemExit(
                f"effective scale {scale} != 1.0 for {base} "
                f"(alpha={alpha} [{alpha_src}], rank={rank}, strength={strength}); "
                "pass --allow-nonunit-scale to override")
        pairs[base + ".weight"] = (A, B, rank, alpha, scale)
    n_tensors = len(sd)
    assert 2 * len(pairs) + len(alpha_keys) == n_tensors, \
        f"unpaired tensors: {n_tensors} total, {len(pairs)} pairs, {len(alpha_keys)} alphas"
    log(f"LoRA: {n_tensors} tensors -> {len(pairs)} pairs; alpha "
        f"{'tensors present' if alpha_keys else 'metadata absent -> convention alpha=rank (scale=1.0)'}")
    return pairs


def compute_delta(name, A, B, scale, groups, head_dim):
    """LoRA delta in DISK layout, fp32."""
    dW = scale * (B.float() @ A.float())
    if is_qkv(name):
        dW = inverse_reorder_continuous_to_grouped(dW, groups, head_dim)
    return dW


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
def build(args, pairs, groups, head_dim, index, shard_names):
    src_tf = os.path.join(args.base, "transformer")
    build_root = args.dst.rstrip("/") + ".building"
    build_fl2va = os.path.join(build_root, "FL2VA")
    build_tf = os.path.join(build_fl2va, "transformer")
    if os.path.exists(build_root):
        raise SystemExit(f"{build_root} already exists — remove it first (never overwrite)")
    if os.path.exists(args.dst):
        raise SystemExit(f"{args.dst} already exists — refusing to overwrite")
    os.makedirs(build_tf)

    all_names = set(index["weight_map"].keys())
    targets = set(pairs.keys())
    missing = targets - all_names
    assert not missing, f"LoRA targets not in checkpoint: {sorted(missing)[:5]}"

    applied = set()
    norm_rows = []
    for shard in shard_names:
        t0 = time.time()
        tensors = load_file(os.path.join(src_tf, shard))
        with safe_open(os.path.join(src_tf, shard), framework="pt", device="cpu") as f:
            meta = f.metadata()
        n_mod = 0
        for name in list(tensors.keys()):
            if name not in targets:
                continue
            A, B, rank, alpha, scale = pairs[name]
            dW = compute_delta(name, A, B, scale, groups, head_dim)
            W = tensors[name]
            assert dW.shape == W.shape, f"{name}: dW{list(dW.shape)} vs W{list(W.shape)}"
            rel = (dW.norm() / W.float().norm()).item()
            norm_rows.append((name, rank, rel))
            tensors[name] = (W.float() + dW).to(torch.bfloat16)
            applied.add(name)
            n_mod += 1
        tmp = os.path.join(build_tf, shard + ".tmp")
        save_file(tensors, tmp, metadata=meta or {"format": "pt"})
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(build_tf, shard))
        log(f"shard {shard}: {n_mod} tensors merged ({time.time()-t0:.0f}s)")
        del tensors

    assert applied == targets, \
        f"applied {len(applied)}/{len(targets)}; missing {sorted(targets-applied)[:5]}"
    log(f"all {len(applied)}/{len(targets)} LoRA targets applied")

    shutil.copy2(os.path.join(src_tf, "config.json"), build_tf)
    shutil.copy2(os.path.join(src_tf, "model.safetensors.index.json"), build_tf)
    shutil.copy2(os.path.join(args.base, "model_index.json"), build_fl2va)
    for comp in ("text_encoder", "video_vae", "audio_vae", "tokenizer", "processor"):
        os.symlink(os.path.join(os.path.abspath(args.base), comp),
                   os.path.join(build_fl2va, comp))
    log("assembled FL2VA dir (transformer real, other components symlinked)")

    norm_rows.sort(key=lambda r: -r[2])
    log("top-10 |dW|_F/|W|_F: " + "; ".join(f"{n}={v:.4f}" for n, _, v in norm_rows[:10]))
    with open(os.path.join(build_root, "delta_norms.csv"), "w") as f:
        f.write("tensor,rank,rel_frobenius\n")
        for n, r, v in sorted(norm_rows):
            f.write(f"{n},{r},{v:.6f}\n")
    return build_root, build_fl2va


# ----------------------------------------------------------------------
# Verification (3 layers)
# ----------------------------------------------------------------------
ORACLE_QKV_KEYS = [
    "blocks.0.attn.qkv_proj.weight", "blocks.10.attn.qkv_proj.weight",
    "blocks.25.attn.qkv_proj.weight", "blocks.49.attn.qkv_proj.weight",
    "token_refiner.blocks.0.attn.qkv_proj.weight",
    "token_refiner.blocks.1.attn.qkv_proj.weight",
]
ORACLE_PLAIN_KEYS = [
    "blocks.0.mlp.fc1.weight", "blocks.0.attn.out_proj.weight",
    "blocks.0.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.weight",
    "blocks.49.mlp.fc2.weight",
]
FORWARD_ORACLE_KEYS = [
    "blocks.0.attn.qkv_proj.weight", "blocks.25.attn.qkv_proj.weight",
    "blocks.49.attn.qkv_proj.weight", "token_refiner.blocks.0.attn.qkv_proj.weight",
    "blocks.0.mlp.fc1.weight", "blocks.49.mlp.fc2.weight",
    "blocks.0.adaln_proj.linear.weight",
]


def get_tensor(root_tf: str, index, name: str) -> torch.Tensor:
    shard = index["weight_map"][name]
    with safe_open(os.path.join(root_tf, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def verify(args, pairs, groups, head_dim, index, shard_names, dst_fl2va: str) -> None:
    src_tf = os.path.join(args.base, "transformer")
    dst_tf = os.path.join(dst_fl2va, "transformer")
    targets = set(pairs.keys())

    # --- layer 1: bit-level recompute over every tensor -------------------
    n_mod = n_same = 0
    for shard in shard_names:
        t0 = time.time()
        orig = load_file(os.path.join(src_tf, shard))
        merged = load_file(os.path.join(dst_tf, shard))
        assert set(orig.keys()) == set(merged.keys()), f"key set differs in {shard}"
        for name, W in orig.items():
            if name in targets:
                A, B, rank, alpha, scale = pairs[name]
                dW = compute_delta(name, A, B, scale, groups, head_dim)
                expected = (W.float() + dW).to(torch.bfloat16)
                assert torch.equal(merged[name], expected), f"bit mismatch: {name}"
                n_mod += 1
            else:
                assert torch.equal(merged[name], W), f"unmodified tensor changed: {name}"
                n_same += 1
        log(f"verify L1 shard {shard} ok ({time.time()-t0:.0f}s)")
        del orig, merged
    total = len(index["weight_map"])
    assert n_mod == len(targets) and n_mod + n_same == total, (n_mod, n_same, total)
    log(f"verify L1 passed: {n_mod} modified + {n_same} unmodified = {total} tensors bit-checked")

    # --- layer 2: independent layout oracle (Comfy single-file BF16) ------
    if args.comfy_single and os.path.exists(args.comfy_single):
        with safe_open(args.comfy_single, framework="pt", device="cpu") as cf:
            comfy_keys = set(cf.keys())
            for key in ORACLE_QKV_KEYS + ORACLE_PLAIN_KEYS:
                assert key in comfy_keys, f"comfy file lacks {key}"
                wc = cf.get_tensor(key)
                wh = get_tensor(src_tf, index, key)
                if is_qkv(key):
                    wh = reorder_grouped_to_continuous(wh, groups, head_dim)
                if not torch.equal(wc, wh):
                    d = (wc.float() - wh.float()).abs().max().item()
                    raise SystemExit(f"verify L2 FAILED on {key}: max_abs_diff={d} "
                                     "(reorder assumption or repack mismatch)")
        log(f"verify L2 passed: Comfy single-file == reorder(HF disk) bit-exact on "
            f"{len(ORACLE_QKV_KEYS)} qkv + {len(ORACLE_PLAIN_KEYS)} plain layers")
    else:
        log("verify L2 SKIPPED (no --comfy-single) — layout not independently confirmed!")

    # --- layer 3: single-layer forward oracle -----------------------------
    gen = torch.Generator().manual_seed(20260808)
    worst_cos, worst_rel = 1.0, 0.0
    for key in FORWARD_ORACLE_KEYS:
        if key not in targets:
            continue
        A, B, rank, alpha, scale = pairs[key]
        W_disk = get_tensor(src_tf, index, key)
        W_merged = get_tensor(dst_tf, index, key)
        if is_qkv(key):
            Wrt = reorder_grouped_to_continuous(W_disk, groups, head_dim).float()
            Wrt_m = reorder_grouped_to_continuous(W_merged, groups, head_dim).float()
        else:
            Wrt, Wrt_m = W_disk.float(), W_merged.float()
        x = torch.randn(16, Wrt.shape[1], generator=gen)
        y_dyn = x @ Wrt.T + scale * ((x @ A.float().T) @ B.float().T)
        y_m = x @ Wrt_m.T
        cos = torch.nn.functional.cosine_similarity(y_m, y_dyn, dim=1).min().item()
        rel = ((y_m - y_dyn).norm() / y_dyn.norm()).item()
        worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
        log(f"verify L3 {key}: min_row_cos={cos:.6f} relL2={rel:.2e}")
        assert cos > 0.9999, f"L3 cosine gate failed on {key}: {cos}"
        assert rel < 5e-3, f"L3 relL2 gate failed on {key}: {rel}"
    log(f"verify L3 passed: worst min_row_cos={worst_cos:.6f}, worst relL2={worst_rel:.2e}")


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lora", required=True)
    p.add_argument("--base", default=os.environ.get("H3_BASE_FL2VA"),
                   help="官方 FL2VA 目录。各机权重根不同,故不设写死默认值;\n                        可用 H3_BASE_FL2VA 环境变量代替")
    p.add_argument("--dst", required=True, help="final model ROOT dir (FL2VA created inside)")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--allow-nonunit-scale", action="store_true")
    p.add_argument("--comfy-single",
                   default=os.environ.get("H3_COMFY_BF16_SINGLE"),
                   help="Comfy single-file BF16 (LoRA training base) for the independent layout oracle")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--lora-revision", default=None)
    args = p.parse_args()
    if not args.base:
        p.error("--base 未给且 H3_BASE_FL2VA 未设:需要指向官方 FL2VA 目录")

    cfg = json.load(open(os.path.join(args.base, "transformer", "config.json")))
    groups, head_dim = cfg["num_attention_heads"], cfg["attention_head_dim"]
    assert groups == 56 and head_dim == 128, (groups, head_dim)
    roundtrip_selftest(groups, head_dim)

    index = json.load(open(os.path.join(args.base, "transformer", "model.safetensors.index.json")))
    shard_names = sorted(set(index["weight_map"].values()))
    log(f"base: {len(index['weight_map'])} tensors in {len(shard_names)} shards; "
        f"heads={groups} head_dim={head_dim}")

    pairs = load_lora_pairs(args.lora, args.strength, args.allow_nonunit_scale)

    if args.verify_only:
        verify(args, pairs, groups, head_dim, index, shard_names,
               os.path.join(args.dst, "FL2VA"))
        log("verify-only: ALL GATES PASSED")
        return

    build_root, build_fl2va = build(args, pairs, groups, head_dim, index, shard_names)
    verify(args, pairs, groups, head_dim, index, shard_names, build_fl2va)

    try:
        script_sha = subprocess.check_output(
            ["sha256sum", os.path.abspath(__file__)], text=True).split()[0]
    except Exception:
        script_sha = "unknown"
    manifest = {
        "base_model": "MiniMaxAI/MiniMax-H3",
        "base_path": os.path.abspath(args.base),
        "base_transformer_index_sha256": sha256_file(
            os.path.join(args.base, "transformer", "model.safetensors.index.json")),
        "lora_repo": "larryvrh/MiniMax-H3-Turbo-Lora",
        "lora_revision": args.lora_revision,
        "lora_file": os.path.basename(args.lora),
        "lora_sha256": sha256_file(args.lora),
        "strength": args.strength,
        "alpha_convention": "absent -> alpha=rank (scale 1.0)",
        "merge_dtype": "float32",
        "output_dtype": "bfloat16",
        "qkv_disk_layout": f"group-interleaved ({groups} groups x [q|k|v] x {head_dim})",
        "qkv_runtime_layout": "Q_all_K_all_V_all (LoRA lora_B order)",
        "merge_script_sha256": script_sha,
        "modified_tensor_count": len(pairs),
        "total_tensor_count": len(index["weight_map"]),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "verification": "L1 bit-level recompute; L2 Comfy single-file layout oracle; "
                        "L3 single-layer forward oracle (cos>0.9999, relL2<5e-3)",
    }
    with open(os.path.join(build_root, "merge_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(build_root, ".complete"), "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    os.rename(build_root, args.dst)
    log(f"DONE -> {args.dst} (manifest + .complete written)")


if __name__ == "__main__":
    sys.exit(main())
