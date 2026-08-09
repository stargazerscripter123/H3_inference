# FINAL · Turbo LoRA 融入 vLLM-Omni / SGLang（6000a）

（2026-08-08/09 by 1ed2dca3 建档。结果全表与 research-note 修正见
`doc/turbo_lora_results.md`；原始记录见本主题 session 目录。）

## 交付结论

**主路线（生产）**：`larryvrh/MiniMax-H3-Turbo-Lora` 的 `v4_step600_ema` 离线
merge 进官方 BF16 FL2VA → 一份 merged checkpoint 两引擎通吃 → 引擎在线 FP8。

| 产线（h3_switch.sh） | 画布 | NFE | warm 中位 | 对照 |
|---|---|---|---|---|
| `vllm-turbo fp8`（生产默认候选） | 864×480 | 6 | **23.5s** | 原 turbo 37.7s → 1.60× |
| 同上（极速档） | 864×480 | 4 | **17.9s** | → 2.11×；仅小运动内容 |
| `vllm-turbo bf16`（无损档） | 864×480 | 6 | 26.2s | 原 vllm 44.2s |
| `sglang-turbo`（768P 高清档） | 1376×768 | 6 | 72.1s | 原 sglang 126.2s → 1.75× |
| `sglang-lora`（dynamic，研究/A-B） | 1376×768 | 6 | 75.2s | VRAM 贴顶 47-48G，非生产 |

Mac CLI：`h3_generate.py --host 6000a --profile {turbo-lora|sglang-turbo} [--nfe N]`
（默认 NFE6；e2e 验证过）。旧 profile 全部不动。

## 关键事实（后续 session 必读）

1. **NFE 语义**：两引擎 `num_inference_steps=N` → N−1 次 forward；引擎传
   `steps=NFE+1` 与作者 sigma 网格逐点相同（三方单测 `test_h3_schedule.py`）。
   客户端 `--nfe` 换算，常数 `NFE_STEP_OFFSET` 在 `run_fl2va_*.py`。上游 sglang
   有开放 PR 要改语义——升级时必须重跑该单测。
2. **配方**：video shift 12 / audio shift 3（引擎默认即 canonical）；strength 1.0；
   alpha 缺省=rank（scale 1.0）。audio shift 6 是旧 ckpt500 实验参数，ablation
   证实对 v4 无增益，已弃。
3. **merge 布局**：LoRA lora_B = runtime 连续 [Q|K|V]/[gate;up]；HF 磁盘 qkv
   为 per-head 交错 → 仅 qkv ΔW 逆 reorder。三层校验（L1 位级 535/535、
   L2 Comfy 单文件独立 oracle 逐位、L3 前向 cos≥0.999999）全过。
   merged 目录含 `merge_manifest.json` + `.complete`（h3_switch 启动前校验）。
4. **runtime LoRA 现状**：vLLM TP4 对 fused 层 warning 后跳过（不可用）；
   SGLang 需 backport 分支 `turbo-lora-backport`（上游 914644e81c9b）+
   显式 `--lora-merge-mode dynamic`（auto 会 merge 进 FP8 权重=禁忌）；
   LoRA 命中 259/266 层为正确值（7 层=LoRA 未打的投影）。
5. **SGLang 本地目录跑 H3**：根目录需完整 HF 布局 + basename 必须 `MiniMax-H3`
   （registry 短名匹配）→ 别名目录 `turbo_v4s600ema_alias/MiniMax-H3`。
6. **质量边界**：8-case Stage A 中 7 个 NFE6 帧级全优；唯一边界=极端运动
   （fight）：v4 NFE4 失败/NFE6 强拖影，**ckpt850 NFE4 保动态明显更好**
   （大运动内容备选；`/v1/set_lora` 可热切）。归因成立：base 同 NFE 单帧被
   fl2va 端点约束托住但运动前载后僵（YDIF 探针）。视频级盲评待用户。
7. **计时口径**：引擎侧固定 seed + warmup1 + timed3（无 ComfyUI 图缓存问题），
   timed 离散 ≤0.1s。base FP8 NFE11 复测 37.7s 与旧档 37.4s 对齐。
8. soak：全日 90+ 单零失败零 OOM，VRAM 逐单恒定。

## 产物索引

- merged：`/home/isaac/Data/h3_weights/MiniMax-H3-Turbo-v4s600ema/`（+62G）
- LoRA：`/home/isaac/Data/h3_weights/loras/`（v4s600ema + ckpt850，revision afc0346）
- 脚本：`merge_turbo_lora.py check_turbo_lora.py test_h3_schedule.py
  test_sglang_lora_slice.py stage_a_run.sh bench_timed.sh ab_ckpt850_dynamic.sh
  smoke_vllm_turbo.sh smoke_sglang_turbo.sh`（6000a scripts/）
- 评测输出：`outputs/stageA_*` `outputs/bench_*` `outputs/ablation_audioshift6`
  `outputs/a2a_vllm_turbofp8_768p` `outputs/turbo_{vllm,sgl}_*`（manifest 齐全）
- gallery：Turbo LoRA 区块 + 大运动边界对比（`gallery/index.html`）

## 未竟

- **Stage B 盲评**（用户参与）：finalist(NFE6 fp8 / NFE4 / ckpt850-大运动) ×
  12-20 case × seed 0/1/2；通过后把 `turbo-lora` 转正为默认 turbo。
- ~~5090 侧 Turbo LoRA~~ **已完成（2026-08-09 by 3195aa2f）**：本地重 merge
  （溯源哈希与 6000a 位级一致）接入 08 主题修复后的 TP2 FP8+DLO 产线；
  NFE6 29.0s / NFE4 21.9s，CLI `--host 5090 --profile turbo-lora`。
  详见 doc/turbo_lora_results.md 第八节 + 本主题 2026-08-09 session notes。
- FP8 rescue（selective BF16 adaln）：本轮 FP8 无回退未触发；`delta_norms.csv` 留档。
- h3_generate.py 任务横幅在 serving-NFE 路径下仍打印 "steps 12"（仅显示瑕疵）。
