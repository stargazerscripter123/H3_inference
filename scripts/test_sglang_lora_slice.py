#!/usr/bin/env python
"""TP unit test for SGLang MergedColumnParallelLinearWithLoRA.slice_lora_b_weights
(backport of upstream 914644e81c9b, PR #33875). CPU-only; run in h3_sglang env.

Checks, for TP=1/2/4 and both H3 fused layer shapes
  qkv sections [7168,7168,7168], fc1 sections [14336,14336], rank 64:
  1. the sliced 2D lora_B equals, per rank, the section-wise rank-local row
     slices concatenated in section order;
  2. forward math: x @ A.T @ B_local.T == the rank's slice of the full LoRA
     delta output (columns of x @ A.T @ B_full.T reordered to rank-local layout);
  3. the 3D (stacked) path still behaves as before.
"""
from __future__ import annotations

import sys

import torch

import sglang.multimodal_gen.runtime.layers.lora.linear as lin


class FakeBase:
    def __init__(self, sections, tp):
        self.output_sizes = list(sections)
        assert all(s % tp == 0 for s in sections)
        self.output_partition_sizes = [s // tp for s in sections]


def make_wrapper(sections, tp):
    w = object.__new__(lin.MergedColumnParallelLinearWithLoRA)
    w.base_layer = FakeBase(sections, tp)
    return w


def reference_slice_2d(B, sections, tp, rank):
    shards, off = [], 0
    for full in sections:
        part = full // tp
        shards.append(B[off + rank * part: off + (rank + 1) * part])
        off += full
    return torch.cat(shards, dim=0)


def main():
    g = torch.Generator().manual_seed(0)
    r = 64
    failures = 0
    for sections in ([7168] * 3, [14336] * 2):
        total = sum(sections)
        B = torch.randn(total, r, generator=g)
        A = torch.randn(r, 5376, generator=g)
        x = torch.randn(8, 5376, generator=g)
        full_delta = x @ A.T @ B.T  # [8, total], columns in full fused order
        for tp in (1, 2, 4):
            for rank in range(tp):
                lin.get_tp_rank, saved = (lambda rk=rank: rk), lin.get_tp_rank
                try:
                    w = make_wrapper(sections, tp)
                    got = w.slice_lora_b_weights(B)
                finally:
                    lin.get_tp_rank = saved
                ref = reference_slice_2d(B, sections, tp, rank)
                ok1 = got.dim() == 2 and torch.equal(got, ref)
                # forward math: rank-local delta must equal the matching columns
                # of the full delta, gathered section-by-section
                cols, off = [], 0
                for full in sections:
                    part = full // tp
                    cols.append(full_delta[:, off + rank * part: off + (rank + 1) * part])
                    off += full
                ref_out = torch.cat(cols, dim=1)
                ok2 = torch.allclose(x @ A.T @ got.T, ref_out, atol=1e-4)
                if not (ok1 and ok2):
                    failures += 1
                    print(f"FAIL sections={sections} tp={tp} rank={rank} "
                          f"slice_ok={ok1} forward_ok={ok2} got_shape={list(got.shape)}")
        # 3D stacked path regression: [n, full, r] -> per-rank [n, part, r]
        n = len(sections)
        B3 = torch.randn(n, sections[0], r, generator=g)
        for tp in (1, 2, 4):
            part = sections[0] // tp
            for rank in range(tp):
                lin.get_tp_rank, saved = (lambda rk=rank: rk), lin.get_tp_rank
                try:
                    w = make_wrapper(sections, tp)
                    got3 = w.slice_lora_b_weights(B3)
                finally:
                    lin.get_tp_rank = saved
                if not torch.equal(got3, B3[:, rank * part:(rank + 1) * part, :]):
                    failures += 1
                    print(f"FAIL 3D sections={sections} tp={tp} rank={rank}")
    if failures:
        print(f"FAILED: {failures} cases")
        return 1
    print("OK: 2D fused slice (qkv 3-section, fc1 2-section) + forward math + 3D path, TP=1/2/4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
