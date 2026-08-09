# Turbo LoRA × vLLM-Omni / SGLang 落地结果（6000a，2026-08-08）

任务：把 `larryvrh/MiniMax-H3-Turbo-Lora` 融入 6000a 两条 serving 产线。
主路线 = **官方 BF16 → 离线 merge → 验证 merged BF16 → 引擎在线 FP8**，一份 merged
checkpoint 两引擎通吃；辅路线 = SGLang runtime dynamic LoRA（研究/A-B 工具）。
档案：`claude_history/07_turbo_lora/`。计划与 feedback 修正见
`~/.claude/plans/okay-project-…misty-jellyfish.md` 与 `doc/temp_research_feedback.md`。

## TL;DR

**计时等级**：`正式` = 固定 seed + warmup1 + timed3 取中位（第六节协议）；`烟测` = 单次或双次，
仅作可行性证据，不可用于对外报数。同一配置以**正式**值为准。

| 产线（6000a） | 画布 | NFE | warm 延迟 | 计时等级 | 对照 base | 状态 |
|---|---|---|---|---|---|---|
| **vLLM merged FP8**（生产候选） | 864×480 | 6 | **23.5s** | 正式 ×3 | base FP8 NFE11 37.7s → **1.60×** | Stage A 过（7/8 case） |
| vLLM merged FP8 极速档 | 864×480 | 4 | **17.9s** | 正式 ×3 | → **2.11×** | 限小运动内容 |
| vLLM merged FP8 质量档 | 864×480 | 8 | 28.9s | 烟测 | → 1.31× | — |
| vLLM merged BF16（无损档） | 864×480 | 6 | 26.2s | 烟测 | vllm BF16 NFE11 44.2s → 1.69× | CLI 未暴露 |
| SGLang merged FP8 高清档 | 1376×768 | 6 | 72.1s | 烟测（单次） | 126.2s → 1.75× | — |
| SGLang dynamic LoRA（研究） | 1376×768 | 6 | 75.2s | 烟测（单次） | — | 259/266 层命中，VRAM 贴顶 |

5090 侧数字见第八节（vllm-turbo NFE6 = 29.0s 正式 ×3 / NFE4 = 22.0s 仅 2 次）。

ComfyUI baseline（INT8 单卡 30 步 ≈100s）对照：**vLLM merged FP8 NFE6 = 4.3×**。

**基线口径**：本文对照一律用 **37.7s**（base FP8 NFE11，与 turbo 同协议重测）。
`doc/speedup_6000a_results.md` 里的 37.4s 是协议对齐前的旧测值，两者差 0.3s 属协议漂移，
不是矛盾；引用时不要混用。

## 一、对 `doc/turbo_lora.md`（research note, 2026-08-07）的修正

1. **audio shift**：v4_step600_ema 的 canonical 配方是 **video 12 / audio 3**
   （= 引擎默认，作者 generate.py 硬编码）。note 里的 "audio shift 6 + 8/10 步"
   属旧 ckpt500 时代社区实验，不适用于 v4；正式弃用前仍安排了 3-vs-6 小 ablation。
2. **推荐 checkpoint**：作者 README 现推荐 `minimax_h3_turbo_v4_step600_ema`
   （NFE 甜区 4–8，>8 过锐，strength 固定 1.0）；4-NFE 大运动备选
   `4step_ema_ckpt850`。note 里的 ckpt500 已过时（上游 sglang cookbook 也还在
   推 ckpt500，以作者 README 为准）。
3. **键名无需任何转换**：LoRA 518 tensors / 259 对，键名已是引擎原生 fused 命名
   （`blocks.N.attn.qkv_proj.lora_A.weight`、`mlp.fc1`、`adaln_proj.linear`、
   token_refiner、final_layer），无前缀。note 担心的"情况 B/C 键名转换、QKV rank
   3r 膨胀"均不存在。alpha 无数值 metadata → 按作者约定 alpha=rank（scale 1.0）。
4. **真正的阻塞在 TP 切片**（note 未覆盖）：
   - vLLM-Omni TP4 runtime LoRA：fused 层（qkv 21504 行 / fc1 28672 行）与
     TP 分片 sum(output_slices) 不匹配，`diffusion/lora/manager.py:697-706`
     **warning 后跳过** → 蒸馏失效，runtime 路线在 TP4 不可用 → 只走 merged。
   - SGLang `MergedColumnParallelLinearWithLoRA.slice_lora_b_weights` 旧版只支持
     3D lora_B，H3 的 2D fused B 会 IndexError；**上游 2026-08-07 已修**
     （commit `914644e81c9b`，PR #33875），本地 cherry-pick 为分支
     `turbo-lora-backport`（4c28e24），TP1/2/4 单测通过。

## 二、步数语义（NFE）——全项目从此用 NFE 表述

两引擎当前 commit（vllm-omni a874b8e09 / sglang 407a65d3）：
`num_inference_steps=N` → linspace N 个 sigma 点 → **N−1 次 DiT forward**。
作者 n 步 = n 次 forward。**引擎传 `steps = NFE+1` 时 sigma 网格与作者
generate.py 逐点相同**（`scripts/test_h3_schedule.py` 三方比对，video shift 12 +
audio shift 3，NFE 4/6/8，max_abs_diff ≤ 8e-8）。

- 客户端 `run_fl2va_{vllm,sglang}.py` 新增 `--nfe`（内部换算，`NFE_STEP_OFFSET=1`
  集中一处）；每个输出旁写 `*.manifest.json`（nfe/steps/shifts/seed/引擎/
  sigma_schedule_sha256/峰值 VRAM/wall）。
- SGLang 有开放 PR 要改成 N 点→N forward；语义若变只改 `NFE_STEP_OFFSET`。
- 服务端日志逐单核对过 forward 数 ≡ NFE（4/6/8 全部精确）。

## 三、merge 管线与三层校验（Gate G1，全过）

`scripts/merge_turbo_lora.py`（6000a）：fp32 累加、完整 scale 公式
`(alpha/rank)×strength`（assert==1.0）、架构参数读 config.json、
`.building` + per-shard fsync + `merge_manifest.json` + `.complete` 原子输出
（h3_switch.sh 启动 turbo 模式前校验 `.complete`）。

**布局要点**：LoRA lora_B 行序 = runtime 连续 [Q|K|V]（Comfy 训练基座布局）；
HF 磁盘 qkv 为 per-head 交错（56 组 × [q|k|v] × 128）→ 仅 qkv 的 ΔW 做逆
reorder，其余 255 对直加；merged 目录保持磁盘布局 → 两引擎加载时各自正 reorder。

| 校验层 | 结果 |
|---|---|
| L1 位级重算 | 535/535 tensors（259 改+276 未改）逐位相等 |
| **L2 独立布局 oracle** | Comfy 单文件 BF16（LoRA 训练基座）== reorder(HF 磁盘) **逐位相等**（6 qkv + 5 平层抽检）——独立于 merge 假设证明 reorder 正确 |
| L3 单层前向 oracle | y_merged vs y_dynamic：worst min_row_cos **0.999999**，relL2 9.7e-4（阈 0.9999 / 5e-3） |

ΔW 分布：`‖ΔW‖F/‖W‖F` 最大在末段 blocks 43-49（≤0.0036），full 表
`merged根/delta_norms.csv`（FP8 rescue 的 ignored-layer 候选依据，暂未用上）。

产物：`/home/isaac/Data/h3_weights/MiniMax-H3-Turbo-v4s600ema/`（+62G，Data 盘余
284G）。LoRA 文件 revision `afc0346`，sha256 见 manifest。

## 四、烟测数字（ToS t=147 标准对，seed 0；正式计时协议另跑）

**vLLM merged（`h3_switch.sh vllm-turbo [bf16|fp8]`，864×480）**

| 精度 | NFE4 | NFE6 | NFE8 | s/it | VRAM/卡 |
|---|---|---|---|---|---|
| BF16 | 25.1s* | 26.2s | 33.5s | 3.30 | ~46G |
| FP8 | 26.5s* | 23.1s | 28.9s | 2.78 | ~38G |

\* 切服务后首单含一次性 compile（首 it 7.7/10.6s）。
**本表是烟测，不作为对外数字**：同配置的正式值见第六节（FP8 NFE6 = 23.5s、NFE4 = 17.9s）。
表中 FP8 NFE4 的 26.5s 是 compile 污染的首单，与正式值 17.9s 不矛盾。

**SGLang merged（`h3_switch.sh sglang-turbo`，1376×768 = 像素 4.3×）**

| NFE4 | NFE6 | NFE8 | s/it | decode |
|---|---|---|---|---|
| 59.6s | 72.1s | 93.1s | 9.9-10.3 | 6.5s |

**SGLang dynamic（`h3_switch.sh sglang-lora`，backport 分支）**：NFE6=75.2s
（10.74 s/it，比 merged 慢 ~8% = 518 个旁路 GEMM）；LoRA 命中 259/266 层
（7 层未覆盖 = LoRA 未打的 patch/time/out 投影，精确符合）；
`/v1/set_lora` 热切 ckpt850 可用。**峰值 VRAM 47-48G 贴顶** → 仅研究用。

质量目检（抽帧）：vLLM BF16/FP8 × NFE4/6/8 与 SGLang merged NFE6 全部无拖影/
端点崩坏/量化伪影；NFE6/8 比 12 步 base 略锐（蒸馏特性）；FP8 与 BF16 同 seed
肉眼无差。

## 五、SGLang 本地目录加载 H3 的两个坑（新台账项）

1. 根目录必须是完整 HF 布局（不止 FL2VA/）→ merged 根把原根所有顶层项 symlink。
2. native pipeline 识别按路径 **basename 短名 == `minimax-h3`** 匹配
   （`KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS`；根 model_index.json 的
   `_class_name` 是 Modular 变体，registry 不认 → 会跌 diffusers 回退崩溃）→
   建别名目录 `turbo_v4s600ema_alias/MiniMax-H3 -> merged根`（h3_switch.sh 自动建）。

## 六、正式计时（固定 seed，warmup1 + timed3；ToS 标准对）

| 配置 | timed×3 | 中位 | 对照 |
|---|---|---|---|
| **vLLM merged FP8 NFE6** | 23.5/23.5/23.6 | **23.5s** | base FP8 NFE11 = 37.7s → **1.60×** |
| vLLM merged FP8 NFE4 | 17.9/17.9/17.9 | **17.9s** | → 2.11× |
| base FP8 NFE11（复现旧产线） | 37.7×7, 37.6 | 37.7s | 旧档 37.4s ✓ 协议对齐 |
| base BF16 NFE20（quality oracle） | ~76s/条 | 76.1s | — |
| SGLang merged FP8 NFE6 @768P | 单次 | 72.1s | base 126.2s → 1.75× |
| **apples-to-apples 1344×768 NFE6** | vLLM 69.7/70.6s vs SGLang 72.1s | — | 同分辨率两引擎打平（vLLM 略快 ~3%） |

计时口径说明：偏离项目旧口径"计时轮换 seed"——那是防 ComfyUI 整图缓存；引擎侧无此
缓存，固定 seed + 每 {分辨率×NFE×精度×ckpt} 组合独立 warmup 才是纯性能对比。
timed×3 离散度 ≤0.1s，引擎调度极稳。

## 六b、Stage A 快筛结果（8 case × 单 seed，帧级 QC + 运动探针；视频级盲评待用户）

素材：镜头切分检测选窗的 ToS 8 case——sniper(标准/枪声) bridge(双人对话)
scope(瞄准镜+大运动) shout(多人+音频事件) holo(全息外观变化) sky(静态复杂背景)
fight(极端快速运动) robot(全身交互)。`inputs/stage_a_final/` + `workflows/stage_a/`。
产出全集 `outputs/stageA_*/`（turbofp8 与 basefp8 各 ×NFE4/6/8[/11]，basebf16 ×NFE20）。

- **7/8 case 在 turbo NFE6 帧级全优**（人物身份/手部/复杂背景/透明全息均无退化，
  略锐于 12 步 base）；NFE8 与 NFE6 相近；NFE4 在小运动 case 亦可用。
- **fight（极端运动）是唯一边界 case**：v4s600 NFE4 融毁失败、NFE6 仍强拖影
  （注意其首尾帧本身即重度运动模糊的电影帧）；**ckpt850 NFE4 明显更好地保留动态
  姿态与运动模糊**（sglang dynamic 热切 A/B，fight/scope/bridge × NFE4/6 共 12 单）
  ——与作者"4 步大运动用 ckpt850"指引吻合。产品侧建议：大运动内容用 ckpt850 或
  NFE≥6，最终以视频盲评定档。
- **归因（base 同 NFE 对照）**：base FP8@NFE4/6 单帧被 fl2va 首尾帧强约束托住
  （不崩），差异在时间维度——YDIF 逐帧运动能量：turbo 均值高 17-25% 且全程持续
  （bridge 四分位 1.89/2.75/3.06/1.62），base 前载后僵（2.51/2.04/1.95/0.99）。
  少步收益确来自 LoRA 而非单纯降步。
- **audio shift 3 vs 6 ablation**（shout/sniper × NFE6）：shift6 无增益
  （RMS 持平或 −1.6dB，无静音/爆音/截断，时长均 5.207s）→ **v4 定档 canonical
  12/3，shift6 分支弃用**（属旧 ckpt500 实验参数；留人工抽听最终确认）。
- **soak 信号（G5）**：turbofp8 连续 36 单零失败、峰值 VRAM 逐单恒定
  （38.0/38.1/38.9/38.0G）；sglang dynamic 12 单 VRAM 稳定；全日合计 90+ 单零 OOM。

## 七、生产建议（待 Stage A/B 确认后定稿）

- `turbo-balanced`＝vLLM merged FP8 / NFE6 / 864×480（生产默认候选，**23.5s** 正式）
- `turbo-fast`＝同上 NFE4（**17.9s** 正式；大运动会拖影 → 见 fight case 与 ckpt850 备选）
- `turbo-quality`＝NFE8 或 BF16 NFE6
- 768P 高清档＝SGLang merged FP8 NFE6（72s；768P 压 30s 需 upscale 路线，本轮不做）
- fallback＝原 turbo（base FP8 12 步）不动；Mac CLI 新增 profile
  `turbo-lora` / `sglang-turbo`（`--nfe` 透传，默认 NFE6），旧 profile 语义不变。

## 八、5090 双卡 TP2 落地(2026-08-09,08 主题 TP2 修复后)

07 未竟的"5090 侧 Turbo LoRA"已闭环:本地重 merge(同 base/LoRA/脚本,
溯源哈希与 6000a 完全一致,L1/L3 校验过)→ 接入 08 主题修复后的
TP2+FP8+DLO 产线(resident=50 全常驻,CUDNN_ATTN,stride 补丁)。

| 配置(2×5090) | NFE | warm | 计时等级 | 对照 |
|---|---|---|---|---|
| vllm-turbo(TP2 FP8+DLO) | 6 | **29.0s**(29.0/29.0/29.1) | 正式 ×3(注:换 seed 1/2/3) | 6000a 同 profile 23.5s;5090 comfy turbo(TeaCache 有损)35s |
| 同上 | 4 | **22.0s**(22.0/21.9) | 仅 2 次 | 6000a 17.9s |

- 峰值 VRAM 23.95G/卡;base NFE11 47s → NFE6 提速 1.62×,与 6000a 的 turbo 增益一致
- 质量:seed0/1/4 帧检查干净(本测试素材为小运动,NFE4 亦可用;大运动场景
  沿用第六节结论:NFE≥6 或 ckpt850)
- 资产:merged 在 5090 `models_merged/MiniMax-H3-Turbo-v4s600ema/`(transformer
  实体 + symlink 回 models_official;manifest+.complete 契约与 6000a 相同)
- 入口:`h3_switch_5090.sh vllm-turbo`;Mac CLI `--host 5090 --profile turbo-lora [--nfe N]`
- 5090 现有三档 serving:comfy turbo 35s(TeaCache)/ vllm base 47s(近无损)/
  **vllm-turbo 29s(NFE6,新最快档)**

### 八b、checkpoint 溯源复核(2026-08-09,补做)

上表数字产出时,`h3_switch_5090.sh` 存在一个**静默服错 checkpoint** 的缺陷:
`vllm)` 与 `vllm-turbo)` 分支都不 `stop_one vllm`,而 `start_vllm` 首行是
"端口健康就 return",所以 vllm ↔ vllm-turbo 直接互切是**空操作**。更麻烦的是
**延迟无法自证**——用两机实测拟合"固定开销 F + 每 forward p":

```text
6000a: F+6p=23.5, F+4p=17.9 -> p=2.8, F=6.7;回代 F+11p=37.5 vs 实测 37.7 ✓
5090 : F+6p=29.0, F+4p=22.0 -> p=3.5, F=8.0;回代 F+11p=46.5 vs 实测 46.9 ✓
=> 5090 上 base 跑 NFE6 也应是 ~29.0s,与 turbo NFE6 实测值重合。
```

加上 `start_vllm` 用 `>` 截断日志,磁盘上无法回溯。**因此对 29.0s 做了独立判定**:
取 5090 的 `outputs/vllm_turbo_5090/Q_nfe6_seed0.mp4`(seed 0,与 6000a 参照同素材同
seed),与 6000a 的两个锚点逐帧比对:

| 比对 | PSNR |
|---|---|
| 5090 Q_nfe6 vs 6000a **turbo** NFE6 | **31.94 dB** |
| 5090 Q_nfe6 vs 6000a **base** NFE6 | 29.70 dB |
| (基准)6000a turbo NFE6 vs base NFE6 = 已知不同 ckpt 的距离 | 30.86 dB |

5090 产物比"已知不同 ckpt"更靠近 turbo、更远离 base;目视亦一致(5090 帧与
6000a turbo 帧同样锐利,base NFE6 明显更糊、面部纹理被抹平)。
**结论:该次 29.0s 用的是 merged Turbo-LoRA,上表数字成立。**

缺陷已修(2026-08-09):`h3_switch_5090.sh` 加 `run/vllm.variant` 变体追踪
(互切必重启)、半死进程清理、端口占用拒启、日志轮转+启动头(记录 model/label/
src HEAD/patch 状态)、`run/vllm.model` 落盘;决策表测试
`scripts/test_h3_switch_5090.sh` 9/9 通过。6000a 的 `h3_switch.sh` 同步加固
(它本就有变体追踪,补的是半死进程与端口占用两条)。

## 九、ComfyUI 原生路线 comfy-int8-turbo-1c(2026-08-09,5090 单卡)

Turbo LoRA 本来就是为 ComfyUI 做的,这条路绕开引擎侧的 merge/TP/DLO 全部复杂度,
**只占一张卡**。用作者的自定义节点 `Larryvrh/ComfyUI-MiniMax-H3-Turbo` @ `55fee864`
(零 pip 依赖),入口 `h3_switch_5090.sh comfy-tlora` → GPU0:8190 独立 worker。

| 配置(1×5090, 864×480, pruned INT8 基座) | NFE | warm wall | denoise | s/it |
|---|---|---|---|---|
| comfy-int8-turbo-1c | 4 | **25.0s** | 14s | 3.53 |
| comfy-int8-turbo-1c | 6 | **30.0s** | 21s | 3.53 |
| comfy-int8-turbo-1c | 8 | 40.0s | 28s | 3.55 |

固定开销约 9-12s(TE 编码 + VAE encode/decode + mux);首单含冷加载 40.2s,不计。

### 横向对照(同机 5090)

| profile | NFE/步 | warm | 占卡 | 近似手段 |
|---|---|---|---|---|
| comfy-int8-original-1c | 30 步 | ~95-110s | 1 | 无 |
| comfy-int8-teacache-1c | 12 步(~7 实算) | 35.0s | 1 | TeaCache 有损缓存 |
| **comfy-int8-turbo-1c** | **6** | **30.0s** | **1** | Turbo LoRA |
| **comfy-int8-turbo-1c** | **4** | **25.0s** | **1** | Turbo LoRA |
| vllm-fp8-turbo-tp2 | 6 | 29.0s | **2** | Turbo LoRA + FP8 |
| vllm-fp8-turbo-tp2 | 4 | 22.0s | **2** | Turbo LoRA + FP8 |

**要点:comfy NFE6 用一张卡做到 30.0s,与 vllm-fp8-turbo-tp2 吃两张卡的 29.0s 打平。**
两张卡各起一个 turbo worker(host RAM 125G / 每 worker ~45G,两个可共存)即可换来
约 2× 吞吐 —— 这是引擎 TP 路线给不了的。该并行配置尚未实测,列为下一步。
同时它比同卡的 TeaCache 档(35.0s)更快**且**无有损缓存。

### 两个会静默毁掉结果的坑(都已踩过并绕开)

1. **不能用 ComfyUI 核心 `LoraLoaderModelOnly`**。`comfy/lora.py` 的
   `model_lora_keys_unet` 只登记 `diffusion_model.<k>` 与 `lora_unet_<k>` 两种形式,
   全树无 MiniMaxH3 分支(已 grep 确认);而 larryvrh 的 LoRA 是裸键
   `blocks.0.attn.qkv_proj.lora_A.weight` → **0/518 命中且不报错**,出片等于
   "6 步的 base"。必须用作者节点 `MiniMaxH3TurboLoRA`(按裸键直接匹配)。
2. **pruned 基座需要运行时补回 time-conditioning**。本机只有 pruned 基座,
   作者节点会用仓库自带的 `h3_silu_temb_grid.safetensors` 注入:日志实证
   `208 backbone modules, 158 bypass adapters, 50 int8 fc2 via merge + 51 adaln
   injected at run time`。drbaph 的 ComfyUI 转换版正是因维度不符丢掉了这 51 个
   adaln 并自带"4 步可能退化或损坏"的警告 —— 走作者节点才拿得回来。

### NFE 语义在 ComfyUI 侧的独立验证

ComfyUI 是 steps==forwards(不像引擎侧 `num_inference_steps=NFE+1`),且
`BasicScheduler` 已硬编码 `scheduler="simple"` —— 正是作者要求的配置,采样器零改动。
实测三档 sigma 网格与作者解析式吻合到小数点后 4 位:

```text
NFE4  实测 1.0, 0.973,  0.9231, 0.8, 0          解析 1.0, 0.972973, 0.923077, 0.8, 0
NFE6  实测 1.0, 0.9837, 0.9601, 0.9231, 0.8575, 0.7064, 0
      解析 1.0, 0.983607, 0.96,  0.923077, 0.857143, 0.705882, 0
NFE8  实测 1.0, 0.9882, 0.973,  0.9524, 0.9231, 0.878, 0.8, 0.6316, 0
```

跨引擎像素比对无意义(同一 seed 数字在 ComfyUI 与 vLLM 产生的初始噪声张量不同,
实测 PSNR 25.9 dB),质量只能做感知比对:NFE4/6/8 三档帧检均连贯,NFE4 略软、
NFE8 略锐,无端点崩坏。

### 客户端接线

`run_fl2va.py` 新增 `--turbo-lora / --lora-strength / --lora-low-vram`;
turbo 开启时同时换用配套的 `MiniMaxH3TurboSampler`。LoRA 节点接在
`UNETLoader(1) → MiniMaxH3TurboLoRA(18) → [TeaCache(17)] → BasicScheduler(8)+BasicGuider(11)`,
LoRA 在前 TeaCache 在后(前者追加权重 patch,后者只挂 wrapper,能把 patch 带下去)
—— 为将来的 `speedup-turbo` 组合档留好了位置。
