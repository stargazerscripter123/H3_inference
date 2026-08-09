# 07_turbo_lora — H3 Turbo LoRA 融入 vLLM-Omni / SGLang（6000a）

## 范围

把 `larryvrh/MiniMax-H3-Turbo-Lora`（few-step 蒸馏 LoRA）落进 6000a 的两条 serving
产线：主路线 = **离线 merge 进官方 BF16 → 引擎在线 FP8**（两引擎共用一份 merged
checkpoint）；辅路线 = SGLang runtime dynamic LoRA（研究/A-B 工具，非生产）。

## 现状指针

- 权威结论：本主题 `FINAL.md`
- 结果文档：`doc/turbo_lora_results.md`（benchmark 表 + 对 `doc/turbo_lora.md`
  research note 的修正）
- merged checkpoint：6000a `/home/isaac/Data/h3_weights/MiniMax-H3-Turbo-v4s600ema/`
  （`merge_manifest.json` + `.complete` 为完整性契约；h3_switch.sh 启动前校验）
- 关键脚本（6000a `~/data/dropbox/CV/h3/scripts/`）：`merge_turbo_lora.py`、
  `check_turbo_lora.py`、`test_h3_schedule.py`、`test_sglang_lora_slice.py`、
  `stage_a_run.sh`、`bench_timed.sh`、`ab_ckpt850_dynamic.sh`

## 主题特有规则

1. **步数一律用 NFE（实际 DiT forward 数）表述**；两引擎当前 commit 语义是
   `num_inference_steps = NFE + 1`（N 个 sigma 点 → N−1 次 forward）。客户端
   `--nfe` 负责换算；绝不把裸 steps 数写进 profile/文档。上游若改语义，改
   `run_fl2va_*.py` 里的 `NFE_STEP_OFFSET`。
2. merge 只允许 `BF16 base + BF16 delta → merged BF16 → 在线量化` 顺序；
   SGLang runtime LoRA 在 FP8 base 上必须显式 `--lora-merge-mode dynamic`。
3. merged-FP8 与 dynamic-FP8 **不要求输出一致**（Q(W+ΔW) ≠ Q(W)+ΔW），只比感知
   质量与速度。
4. 性能计时固定 seed、warmup 1 + timed ≥3 取中位（引擎无 ComfyUI 图缓存，
   与项目旧口径"计时轮换 seed"不同，原因见结果文档）；质量评测才换 seed。
