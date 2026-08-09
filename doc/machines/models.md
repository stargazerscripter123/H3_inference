# 跨机权重台账

**每份权重是什么、从哪来、在哪几台机器上、怎么校验、占多少盘。**
事实来自三份机器档案(2026-08-09 各机实测),此处只做汇总与跨机对照;
路径/命令的完整上下文见 [`popos-5090.md`](popos-5090.md) / [`popos-6000a.md`](popos-6000a.md) /
[`runpods-5090x4.md`](runpods-5090x4.md) 的第 5 节与第 7 节。

**本仓库不含任何权重**(合计数百 GB,`.gitignore` 已排除 `models*/ base/ merged/ loras/`)。

---

## 0. 先看这个:什么能当「权重身份」,什么不能

这是本文最容易被误用的一节。判据分三档:

| 判据 | 能证明什么 | 能不能当门禁 |
|---|---|---|
| HF repo 的 **snapshot sha / `refs/main`** | **几乎什么都不能证明** | ❌ **不行**。三机各不相同(`b3c7290e…` / `101ecd0a…` / `bfc8ed03…`),而上游 `MiniMaxAI/MiniMax-H3` 在采集当天(08-09)10:05 又前进到 `6818f6c3…` —— **一天内动了两次**。它只是「某次下载的记录」,用来重现同一次下载 |
| `FL2VA/transformer/model.safetensors.index.json` 的 **sha256** | **张量名 / 分片划分 / 总字节**三者一致 | ✅ 是门禁,但**不等于位级等价**。实测该文件只有两个 key:`metadata`(仅 `total_size: 66280430144`)与 `weight_map`(535 条映射)——**里面没有任何分片摘要**。上游任何一次「保持张量名/形状/分片/总字节不变」的重传都会产出字节完全相同的 index |
| **13 个分片自身的 sha256** | **位级等价**(唯一能证明的东西) | ✅ **真门禁,但目前三机都未采集** —— 见 §6 待补 |

> ⚠️ **「三机权重位级等价、可互引 benchmark」目前只是高置信推测,不是已证事实。**
> 现有证据:三机 index sha256 相同 + merge manifest 的输入哈希相同 + 各机 merge 自校验 L1 通过
> (259 改 + 276 未改 = 535 逐位重算)+ 5090 与 6000a 的成片 PSNR 复核落在预期档
> (31.94 dB vs 「已知不同 ckpt」基准 30.86 dB)。**缺的正是分片 sha256 那一项。**

---

## 1. 总表

体积列:精确字节数来自各机 `stat -c %s` 实测;`du -sh` 口径的用 G 标注。

> ⚠️ **单位口径**:本节括号里由字节数换算的 `G` 是 **GB(÷10⁹)**,而 §4 磁盘预算表与
> `README.md` §1.6 用的是 `du -sh` 的 **GiB(÷2³⁰)**。同一个文件因此会出现两个数字,
> 不是两份权重:#2 Turbo LoRA = 0.78 GB = **0.74 GiB**;#11 BF16 单文件 DiT = 66.3 GB = **62 GiB**。
「在哪台机」列:✅ 有 / — 无。各机路径见 §2(按 # 对应)。

| # | 权重 | HF repo @ revision | 精度 | 体积 | 5090 | 6000a | runpods | 服务哪些 profile | 获取方式 | 校验契约 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **官方基座 FL2VA**(整仓下,大件在 `FL2VA/`) | `MiniMaxAI/MiniMax-H3` @ 各机不同(见 §0 与 §2 注) | BF16 分片 | **135G**(`FL2VA` 本体;含下载残留时 145G/168G) | ✅ | ✅ | ✅ | 全部 `vllm-bf16-*` / `vllm-fp8-*`(在线量化)/ `sglang-*`;**merged 的 base** | `hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/**" --local-dir …`(`--include` 必须重复写);或局域网 rsync | `FL2VA/transformer` **13 个 shard** + `text_encoder` 14 个 shard;index sha256 = `fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb`(**三机一致**) |
| 2 | **Turbo LoRA(生产)**<br>`minimax_h3_turbo_v4_step600_ema.safetensors` | `larryvrh/MiniMax-H3-Turbo-Lora` @ **`afc0346516372a17162c14df3c5264de1d9aa1c0`** | BF16 | **779 849 816 B**(744M) | ✅ | ✅ | ✅ | merge 的输入;`comfy-int8-turbo-1c`;`h3_switch.sh sglang-lora` | `hf download … --include "<单文件>"`(**整仓 22 个文件 = 白下 22G**) | **sha256 = `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`**(三机一致);G0 体检 `scripts/check_turbo_lora.py <lora路径>` → 259 对键 / 518 tensors / 全 BF16 |
| 3 | Turbo LoRA(大运动备选)<br>`minimax_h3_turbo_4step_ema_ckpt850.safetensors` | 同 repo 同 revision | BF16 | 779 849 816 B | — | ✅ | — | A/B 研究,不在 profile 矩阵 | 同上 | sha256 = `5a6eeba171cf183020a4ad48774bb2968f29f8168afd6ec17a04987f3528b4ea` |
| 4 | **merged Turbo checkpoint**<br>`MiniMax-H3-Turbo-v4s600ema` | **不是下载来的**,本地 merge 产出(见 §3) | BF16 | **62G**(`FL2VA/transformer` 13 个真实分片,其余顶层项 symlink 回 base) | ✅ | ✅ | ✅ | 全部 `vllm-*-turbo-*` / `sglang-fp8-turbo-tp4` | `scripts/merge_turbo_lora.py`(§3) | 根目录 **`.complete`** + `merge_manifest.json` + `delta_norms.csv`(260 行);switch 脚本缺 `.complete` **拒绝启动** |
| 5 | ComfyUI pruned **INT8** DiT<br>`minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `Comfy-Org/MiniMax-H3` `resolve/main`(**未记 revision**) | INT8 convrot | **20 970 379 616 B** | ✅ | ✅(实体在根盘) | — | `comfy-int8-original-1c` / `-teacache-1c` / `-turbo-1c`(两机) | `bash scripts/download_h3.sh {5090\|6000a}` | **精确字节数** + `curl -C -` 断点续传 |
| 6 | ComfyUI pruned **FP8_scaled** DiT | `Comfy-Org/MiniMax-H3` | FP8 scaled | **20 958 205 608 B** | ✅ | — | — | **仅 A/B 对照**,不进生产(实测比 INT8 慢 ~20%:3.96 vs 3.29 s/it) | `bash scripts/dl_fp8_auth.sh`(带 token 的 curl,**为规避限速不是鉴权**) | 精确字节数 |
| 7 | TE **NVFP4-AWQ**<br>`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `Comfy-Org/MiniMax-H3` | NVFP4 AWQ | **15 687 142 551 B** | ✅(**5090 默认 TE**) | — | — | 5090 三条 ComfyUI 产线 | `download_h3.sh 5090` | 精确字节数 |
| 8 | TE **INT8-convrot**<br>`qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `Comfy-Org/MiniMax-H3` | INT8 convrot | **27 141 342 152 B** | ✅(备用) | ✅(**6000a 默认 TE**,实体在根盘) | — | 6000a ComfyUI 产线 | `download_h3.sh {5090\|6000a}` | 精确字节数 |
| 9 | video VAE<br>`minimax_h3_video_vae_fp16.safetensors` | `Comfy-Org/MiniMax-H3` | **FP16** | **5 207 808 496 B** | ✅ | ✅(根盘) | — | 全部 ComfyUI profile | `download_h3.sh` | 精确字节数 |
| 10 | audio VAE<br>`minimax_h3_audio_vae_fp32.safetensors` | `Comfy-Org/MiniMax-H3` | **FP32(硬性要求)** | **605 254 808 B** | ✅ | ✅(根盘) | — | 全部 ComfyUI profile | `download_h3.sh` | 精确字节数;⚠️ **换 BF16 会让音量 −20 dB,而且不报错** |
| 11 | ComfyUI **BF16 单文件 DiT**<br>`minimax_h3_fl2va_bf16.safetensors` | `Comfy-Org/MiniMax-H3` | BF16 | **66 280 487 368 B**(66.3G) | ❌(想跑 L2 需补下) | ✅ | — | `comfy-bf16-original-4c`(6000a 4 卡 oracle);**merge 的 L2 布局 oracle** | `bash scripts/download_bf16.sh`(**6000a 专用**)或直接 `curl -L -C -` | 精确字节数 |
| 12 | ComfyUI **BF16 单文件 TE**<br>`qwen3vl_32b_minimax_h3_bf16.safetensors` | `Comfy-Org/MiniMax-H3` | BF16 | **51 506 295 256 B**(51.5G) | — | ✅ | — | `comfy-bf16-original-4c` | `download_bf16.sh` | 精确字节数 |
| 13 | TE FP8(SGLang 路线)<br>`Qwen3-VL-32B-Instruct-FP8` | `Qwen/Qwen3-VL-32B-Instruct-FP8`;HF cache `refs/main` = `4bf2c2f39c37c0fede78bede4056e1f18cdf8109`(**只有 ref 没有 blob**) | FP8 | **34G**,7 分片 | ⚠️ 有,但**路线已死** | — | — | `h3_switch_5090.sh sglang` 的 `--text-encoder-path` | **走 ModelScope**(HF 限速卡死过);vLLM 路线在线量化**不需要它** | 无字节校验脚本;判据是 `logs/download_te_ms.log` 末尾 `MS_DL_EXIT=0` |
| 14 | ModelOpt 混合 FP8 | `feizhai123/MiniMax-H3-ModelOpt-Mixed9-Dynamic-FP8` @ `80a8efc5bb9473f12ec1f0e5a1b20be5c7765fe7` | FP8 per-channel-per-token 混合 | **5.0G,残缺**(38/74 文件;真实落地仅约 600M,其余 4.4G 在 `.cache/` 里是未完成分块) | ⚠️ 有 | — | — | **无人使用**,没有任何脚本引用 | — | — |
| 15 | HF cache 桥接(不含实体权重) | — | — | 676K / 24M | ✅ | ✅ | ❌ | 让 SGLang 用仓库名命中本地文件,免重下 145G;**vLLM/ComfyUI 都不需要** | `cp -rs` 复刻结构 + 写 `refs/main`(见各机第 7 节) | 结构:顶层真实目录 + 目录内逐文件 symlink |

> ⚠️ **#1 的 revision 三机不同,#4 的产出时间三机不同 —— 这两条都是预期的,不是故障。** 见 §0 与 §3.4。
> ⚠️ **`Comfy-Org/MiniMax-H3` 的单文件全都没记 revision**(`resolve/main` 直链),身份判据**只有字节数**。
> 上游若无声重传,字节数不变就发现不了。这是已知的弱环。

---

## 2. 各机路径速查

| # | popos-5090 | popos-6000a | 5090-Runpods |
|---|---|---|---|
| 1 | `models_official/MiniMax-H3/` | `/home/isaac/Data/h3_weights/MiniMax-H3/` | `/workspace/h3/base/MiniMax-H3/` |
| 2 | `loras/…v4_step600_ema.safetensors`(+ symlink 进 `ComfyUI/models/loras/`) | `…/h3_weights/loras/` | `/workspace/h3/loras/MiniMax-H3-Turbo-Lora/` |
| 3 | — | `…/h3_weights/loras/` | — |
| 4 | `models_merged/MiniMax-H3-Turbo-v4s600ema/` | `…/h3_weights/MiniMax-H3-Turbo-v4s600ema/`(+ SGLang 别名 `…/turbo_v4s600ema_alias/MiniMax-H3` → 它) | `/workspace/h3/merged/MiniMax-H3-Turbo-v4s600ema/` |
| 5–10 | `ComfyUI/models/{diffusion_models,text_encoders,vae}/`(**全是实体文件**) | INT8 两件 + 两个 VAE **实体在根盘**同名目录;BF16 两件是 **symlink → 第二块 NVMe** | — |
| 11–12 | — | `…/h3_weights/{diffusion_models,text_encoders}/` | — |
| 13 | `models_te_fp8/Qwen3-VL-32B-Instruct-FP8/` | — | — |
| 14 | `models_modelopt/MiniMax-H3-FP8/` | — | — |
| 15 | `~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/` | 同左 | — |

**各机 #1 的 revision 与来源(三条都不一样,溯源时按这个查)**:

| 机器 | HF cache `refs/main` 写的 | 磁盘字节的真实来源 | 备注 |
|---|---|---|---|
| popos-5090 | `b3c7290e66afdf293bef3b9077b7a266ef421f34` | **不确定** | 证据打架:`logs/` 里没有完整下载日志(指向 rsync),但 `.cache/huggingface/` 里躺着 11G xet 半成品(指向本机 `hf download`)。最能解释两组证据的假说是「先下到一半,再用 rsync 补齐」——**仍是推测**。对复现无影响 |
| popos-6000a | `b3c7290e66afdf293bef3b9077b7a266ef421f34` | **`101ecd0aa25532b6443c625340f095a617a2526a`** | ⚠️ **桥接宣称的与磁盘实际的不是同一个**:138 个 `.metadata` 文件第一行的 commit **全部**是 `101ecd0a…`。功能无碍(SGLang 只按 `refs/main` → snapshot 目录解析),但**溯源以 `101ecd0a…` 为准**,复现请用它 |
| 5090-Runpods | `bfc8ed0353f5a9733be73e6b2c98ec0948195b86` | 同 ref(`--local-dir` 直下) | — |

⚠️ **三机 revision 各不相同,但 `FL2VA/transformer` 的 index sha256 完全相同。** 这正是 §0 那条
「snapshot sha ≠ 权重身份」的现场。

---

## 3. merged Turbo 权重(#4)——它不是下载来的

### 3.1 合并输入与产出

```text
输入 A: 官方基座 #1 的 FL2VA/            (BF16,535 tensors / 13 shards,heads=56 head_dim=128)
输入 B: Turbo LoRA #2                    (BF16,518 tensors -> 259 对键,alpha 元数据缺省)
       ↓  scripts/merge_turbo_lora.py    (merge_dtype=float32, output_dtype=bfloat16)
产出  : MiniMax-H3-Turbo-v4s600ema/      (62G;FL2VA/transformer 13 个真实分片,
                                          其余顶层项 symlink 回 base;根目录写 .complete)
```

命令(以 6000a 为例,其余两机只是路径不同):

```bash
$VLM_PY scripts/merge_turbo_lora.py \
  --lora   <#2 的路径> \
  --lora-revision afc0346516372a17162c14df3c5264de1d9aa1c0 \
  --base   <#1>/FL2VA \
  --dst    <目标>/MiniMax-H3-Turbo-v4s600ema \
  --strength 1.0 \
  --comfy-single <#11 的路径>          # 没有 #11 时必须显式写 /nonexistent,见 3.3
```

> ⚠️ **两个脚本的参数形式不一样,别混**:
> `check_turbo_lora.py <lora>` 是**位置参数**(写 `--lora` 会被 argparse 直接拒绝);
> `merge_turbo_lora.py --lora <lora>` 是选项。
> ⚠️ **为什么必须 merge 而不是运行时挂 LoRA**:vLLM-Omni 的 runtime LoRA 对 fused 层
> (qkv 21504 行 / fc1 28672 行与 `sum(output_slices)` 对不上)只 **warning 后跳过**,
> 蒸馏**静默失效**。dynamic LoRA 只在 SGLang 上做。
> ⚠️ **为什么 strength 1.0 / alpha 缺省**:LoRA 文件里没有 alpha 元数据,约定 `alpha = rank`(scale 1.0)。

### 3.2 `merge_manifest.json` 里该有什么

| 字段 | 值 / 含义 | 三机是否一致 |
|---|---|---|
| `base_model` | `MiniMaxAI/MiniMax-H3` | ✅ |
| `base_path` | 各机 base 的绝对路径 | ❌ 机器相关,正常 |
| `base_transformer_index_sha256` | `fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb` | ✅ **门禁** |
| `lora_repo` / `lora_file` | `larryvrh/MiniMax-H3-Turbo-Lora` / `minimax_h3_turbo_v4_step600_ema.safetensors` | ✅ |
| `lora_revision` | `afc0346516372a17162c14df3c5264de1d9aa1c0`(runpods 记短名 `afc0346`,同一 commit) | ✅(记法不同) |
| `lora_sha256` | `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3` | ✅ **门禁** |
| `strength` / `alpha_convention` | `1.0` / `absent -> alpha=rank (scale 1.0)` | ✅ |
| `merge_dtype` / `output_dtype` | `float32` / `bfloat16` | ✅ |
| `qkv_disk_layout` | `group-interleaved (56 groups x [q\|k\|v] x 128)` | ✅ |
| `qkv_runtime_layout` | `Q_all_K_all_V_all (LoRA lora_B order)` | ✅ |
| `modified_tensor_count` / `total_tensor_count` | **259 / 535** | ✅ **门禁** |
| `created` | 各机合并时刻 | ❌ 不同,正常 |
| `merge_script_sha256` | 三机都记 `b034c41b1c20233333890d5c168583de48730fa9f7df10c2c4041c615b5dcb1b` | ⚠️ **不是门禁,见下** |
| `verification` | 一句**静态模板**字符串 | ⚠️ **不可信,见 3.3** |

> ⚠️ **`merge_script_sha256` 对任何新人都必然对不上,这是已知情况,不是合并出错。**
> 该字段是**脚本自身文件**的 sha256(`merge_turbo_lora.py:349` 算,`:367` 写)。
> manifest 记的 `b034c41b…` 那一版脚本**已不存在于任何机器、也从未进过 git 历史**
> (`git log -- scripts/merge_turbo_lora.py` 只有一次初始导入 `d66049a`;全树只有一份该文件;
> 其 mtime 晚于三次 merge 发生的时间 = 合并之后脚本被覆盖过)。
> 仓库现版本(Mac / 6000a / runpods 三处一致)= `70cdfe9d7b1e1dd837e61fc4fea875e383c223aaf311dd87f3f22dbe60ba777d`。
> **用现版本重跑,该字段必然写成 `70cdfe9d…` —— 属预期,已永久排除在门禁之外。**

### 3.3 `.complete` 契约,以及 L1/L2/L3 三层校验

**`.complete` 的意义**:它是「这份 62G 目录不是半成品」的**唯一凭据**,写在**根目录**
(不是 `FL2VA/` 里)。三份 switch 脚本启动 turbo 前都检查它,缺了就**拒绝启动** ——
这挡的是「merge 跑了一半被 kill,引擎照样把残缺分片当权重服出去」这类静默事故。
内容是合并完成的时间戳:

| 机器 | `.complete` 内容 | merge 耗时 |
|---|---|---|
| popos-6000a | `2026-08-08T15:58:07+1000` | ~3 min(13 shard × ~9s 合并 + ~5s/shard L1) |
| popos-5090 | `2026-08-09T08:48:22+1000` | ~2.5 min(merge ~90s + L1 ~56s) |
| 5090-Runpods | `2026-08-09T01:45:43+0000` | ~9 min(CPU 较弱) |

**三层校验**:

| 层 | 验什么 | 三机结果 |
|---|---|---|
| **L1** 位级重算 | 逐 shard 重算并逐位比对:`259 modified + 276 unmodified = 535 tensors bit-checked` | ✅ 三机全过 |
| **L2** 独立布局 oracle | Comfy 单文件 BF16(LoRA 的训练基座)`== reorder(HF 磁盘)` 逐位相等(抽检 6 个 qkv + 5 个平层)。**独立于 merge 自身假设**,证明 qkv 的 group-interleaved 磁盘布局 ↔ `Q_all/K_all/V_all` 运行时布局的重排是对的 | ✅ **只有 6000a 真跑过**(它是唯一同时有 HF 分片和 Comfy 单文件 #11 的机器);5090 与 runpods **SKIPPED** |
| **L3** 单层前向 oracle | `y_merged` vs `y_dynamic`:`worst min_row_cos = 0.999999`,`relL2 = 9.65e-04`(阈值 0.9999 / 5e-3) | ✅ 三机全过,数值完全相同 |

> ⚠️⚠️ **这是全文最容易被照抄坏的一处。**
> `merge_turbo_lora.py:264` 是 `if args.comfy_single and os.path.exists(args.comfy_single):` ——
> **`--comfy-single` 指向的路径不存在时,L2 静默跳过**,日志里只留一行
> `verify L2 SKIPPED (no --comfy-single) — layout not independently confirmed!`,
> **退出码仍是 0**,而 manifest 的 `verification` 字段照写「L2 Comfy single-file layout oracle」。
>
> **判据只有一个:`logs/merge*.log` 里必须出现 `verify L2 passed`。**
> 看到 `verify L2 SKIPPED` 就说明**布局从未被独立确认过** —— 那不是「输出正常」。
> **手上没有可比对机器时,请务必先下 #11(66.3G)让 L2 真跑。**

### 3.4 怎么验证两台机器合出来的是逐位一致的

**现状:严格意义上还没验过。** 现有的替代链是「证明输入相同」,不是「证明输出相同」:

| 已有证据(三机一致) | 强度 |
|---|---|
| `base_transformer_index_sha256` 相同 | 只证张量名/分片划分/总字节一致(§0) |
| `lora_sha256` 相同 | LoRA 输入位级相同 ✅ |
| `modified/total = 259/535` 相同 | 命中的 LoRA target 数一致 |
| 各机 L1 自校验通过 | 证明「本机 merge 的算术是自洽的」,不跨机 |
| 6000a 的 L2 通过 | 证明布局解释正确 —— 但**只在 6000a 上验过** |
| ~~`merge_script_sha256` 相同~~ | ❌ **已作废**(见 3.2),不再是链条的一环 |

**要真正断言位级等价,做这一步**(62G 全盘读,挑两台机器都空闲时做):

```bash
# 在两台机器上各跑一次,逐行对比 13 行输出
cd <merged>/MiniMax-H3-Turbo-v4s600ema/FL2VA/transformer && sha256sum model-000*-of-00013.safetensors
# 顺带把 base 的 13 个分片也算一遍,补上 §0 缺的那个真 pin
cd <base>/MiniMax-H3/FL2VA/transformer   && sha256sum model-000*-of-00013.safetensors
```

采集后写回 `merge_manifest.json` 与本文 §1。**在补上之前,「三机可互引 benchmark」是推测。**

**间接旁证(已做,不能替代上面那步)**:5090 与 6000a 的成片逐帧 PSNR 复核 ——
`5090 Q_nfe6 vs 6000a turbo NFE6 = 31.94 dB`,`vs 6000a base NFE6 = 29.70 dB`,
而「已知不同 ckpt」的基准距离是 30.86 dB。**这证明 5090 那次 29.0s 服的确实是 merged Turbo,
不是 base** —— 但它证明不了两份 checkpoint 逐位相同。

### 3.5 SGLang 专用别名(只有 6000a 需要)

```bash
mkdir -p "$STORE/turbo_v4s600ema_alias"
ln -s "$STORE/MiniMax-H3-Turbo-v4s600ema" "$STORE/turbo_v4s600ema_alias/MiniMax-H3"
```

> ⚠️ 别名的 **leaf 必须叫 `MiniMax-H3`**:SGLang 的 `KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS`
> 按路径 basename 短名匹配;根 `model_index.json` 的 `_class_name` 是 `MiniMaxH3ModularPipeline`,
> registry 不认,会跌回 diffusers 路径然后崩。`h3_switch.sh` 会自动建这个 symlink。
> merged 根还**必须是完整 HF 布局**(把 base 根的所有顶层项 symlink 过来),不能只有 `FL2VA/`。

---

## 4. 磁盘预算

### 4.1 按机器角色分档

| 角色 | 需要哪几项(§1 的 #) | 稳态 | 下载期额外 | **建议可用空间** |
|---|---|---|---|---|
| **纯引擎 serving**(runpods 型) | #1 135G + #4 62G + #2 0.8G | **≈ 198G** | +56G(xet 残留)+ venv 12G | **≥ 280G** |
| **serving + ComfyUI 量化产线**(5090 型) | 上面 + #5 #7 #8 #9 #10(85G,含可选 #6) | **≈ 285G** | +11G | **≥ 320G**(想留余量按 420G) |
| **全功能机**(6000a 型,含 BF16 4 卡 oracle) | 上面 + #11 #12(118G) | **≈ 376G** | +≥15G | **≥ 450G** |
| 追加:让 merge 的 **L2 真跑** | 只需 #11 | +62G | — | 见 §3.3,**强烈建议** |
| 追加:复现已判死的 SGLang TE FP8 | #13 | +34G | — | 一般不需要 |

### 4.2 各机实测拆解

| | popos-5090 | popos-6000a | 5090-Runpods |
|---|---|---|---|
| #1 官方基座 | 145G(FL2VA 135G + `.cache/` **11G**) | 145G(同左) | 135G(另有 `.cache/` **34G**) |
| #4 merged | 62G | 62G | 62G |
| ComfyUI 单文件 | 85G(#5–#10 共 6 个) | 51G(#5 #8 #9 #10,**实体在 94% 满的根盘**) | — |
| #11 #12 BF16 单文件 | — | 118G(第二块 NVMe) | — |
| LoRA | 0.74G | 1.5G(#2 + #3) | 0.74G(另有 `.cache/` **22G**) |
| 其他 | #13 34G + #14 5G | — | venv 12G |
| **合计** | ~332G | ~376G | 263G |
| **其中可回收** | **~50G**(#13 34G + #14 5G + `.cache` 11G) | 11G(`.cache` 里 8 个 `.incomplete`) | **56G**(base 34G + LoRA 22G) |
| **生产真正需要** | **~283G** | ~376G | **197G** |

> ⚠️ **6000a 两块盘都已 93–94% 满**,且 `ComfyUI/models` 的 51G 实体落在根盘、又在 Dropbox 同步目录里。
> 新增权重一律往大盘放并 symlink 回去(`download_bf16.sh` 已这么做,`download_h3.sh` **没有**)。
> 复现时**建议把 `ComfyUI/models` 整体 symlink 到大盘**,避免重蹈覆辙。
> ⚠️ **runpods 的 `/` 是临时盘,pod 重建即失**;只有 `/workspace` 活得下来。
> ⚠️ 清残留前**先确认没有下载正在进行**:
> `find <dir>/.cache -name "*.incomplete" -printf "%s %p\n" | sort -n`,确认后再删。
> (本轮采集在严格只读约束下,**三台机器一样都没删**。)

---

## 5. 认证与下载

### 5.1 门禁状态(实测 HF API,2026-08-09)

| repo | `gated` | `private` | 需要 token 吗 | 实际取法 |
|---|---|---|---|---|
| `MiniMaxAI/MiniMax-H3` | false | false | **不是权限需要,是限速需要** | `hf download`(带 token) |
| `larryvrh/MiniMax-H3-Turbo-Lora` | false | false | 同上 | `hf download --include <单文件>` |
| `Comfy-Org/MiniMax-H3` | false | false | **不需要** | `download_h3.sh` / `download_bf16.sh` 走**裸 `curl -sSL --fail -C -`,全程无 `Authorization` 头** |
| `Qwen/Qwen3-VL-32B-Instruct-FP8` | — | — | HF 侧限速卡死过 | **换 ModelScope**(19 文件 / 2h13m) |

> 三个主 repo 当前**都不是 gated**,所以 401/403 基本只会是「token 没写对」或「限速」,不是门禁。
> 但上游随时可能改成 gated —— 真遇到 403,先去网页看是否需要 accept license,再 `hf auth login`。

### 5.2 token 放哪(**绝不要把 token 的值写进任何文档**)

| 机器 | token 文件 | 备注 |
|---|---|---|
| popos-5090 / popos-6000a | `~/.cache/huggingface/token` | 环境里**没有** `HF_*` 变量,不走 env。5090 实测 37 字节;**6000a 只确认了文件存在,未读大小/内容** |
| 5090-Runpods | `/root/hf-cache/token`(`HF_HOME`)+ `/root/.cache/huggingface/token` | ⚠️ 两个都在**临时盘**,pod 重建即失,由重建流程重新写入 |

**token 的实际值只在 `credentials/HF.md`,该目录已被 `.gitignore` 排除,不入库。**
写入方式:`hf auth login`,或 `printf %s '<token>' > ~/.cache/huggingface/token && chmod 600 …`。

> ⚠️ **token 必须在下载脚本之前写好**:`scripts/dl_fp8_auth.sh:6` 是
> `TOKEN=$(cat ~/.cache/huggingface/token)`,`set -u` 管不到命令替换失败 ——
> 没有 token 时它**不报错退出**,而是带着**空 Bearer 头**空转
> (`for i in $(seq 1 40)` × `--retry 3`),要转很久才打印 `FP8_DOWNLOAD_FAILED`。**极难排查。**

### 5.3 匿名下载会限速

- 匿名走 HF 会撞到**日出口约 300G** 的配额,历史上 144G 的基座下载卡住过。
- Qwen TE FP8(#13)在 HF 上直接卡死(卡在 13G),最终**改用 ModelScope** 才下完
  (`modelscope` / `modelscope-hub` 两个包因此进了 comfy 与 sglang 环境)。
- 所以:**token 不是为了过权限,是为了不被限速。**

### 5.4 五个静默陷阱(踩一个就白下几十上百 G)

| # | 陷阱 | 正确写法 |
|---|---|---|
| 1 | `HF_HUB_ENABLE_HF_TRANSFER=1` **已被 huggingface_hub 弃用**,走退化路径只有 ~28MB/s(日志里有 FutureWarning 原文)。`setup_runpods.sh` / `dl_fl2va_auth.sh` 至今还在用 —— **已知待修缺陷** | `export HF_XET_HIGH_PERFORMANCE=1`(峰值实测 247MB/s) |
| 2 | **多个 `--include` 空格并列会被当成位置参数 `FILENAMES`**,于是打印 `Ignoring --include since filenames have been explicitly set.` 然后 **`--include` 整个被忽略,135G 一个文件都不下** —— 而这只是一条 warning | `--include A --include B --include C`(可重复的单值选项) |
| 3 | **LoRA 仓有 22 个实验 checkpoint,整仓拉 = 白下 22G**(runpods 就是这么踩的,那 22G 残留至今还在盘上) | 必须 `--include "<单个文件名>"` |
| 4 | **只 `--include "FL2VA/**"` 会漏掉根级 `model_index.json`**,而 SGLang 的 `--model-path MiniMaxAI/MiniMax-H3` 正是靠它解析仓库 | 连根级 json / README 一起下;或整仓下(多约 60M) |
| 5 | 反过来,想「只补根级小文件」时**只排 `FL2VA/` 和 `Ref2VA/` 不够** —— 该 repo 顶层 `transformer/` `transformer_ref/` `text_encoder/` `vae/` 里**各自还有整套 safetensors 分片**(46 个),会额外拉几百 GB | `--exclude "FL2VA/*" --exclude "*.safetensors" …`,先 `--dry-run` 看清单 |

其他:
- **`huggingface-cli` 已被移除**(1.26.0 下只打一句 deprecated 然后拒绝执行)。命令名是 **`hf`**,
  由 `pip install "huggingface_hub[cli]"` 提供。
- 两台本地机上 `hf` 装在 `h3_comfy_NV_py312` 里,而复现步骤全程不 `conda activate` ——
  **必须用绝对路径** `$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/hf`,裸敲会 `command not found`。
- `hf download --local-dir` 会在目标目录下留一个 `.cache/huggingface/`(见 §4.2 的可回收项)。

---

## 6. 待补 / 已知弱环

| # | 事项 | 影响 |
|---|---|---|
| 1 | **13 个 transformer 分片的 sha256 未采集**(base 与 merged 各一份) | 「三机权重位级等价」缺唯一的真证据(§0 / §3.4) |
| 2 | **`Comfy-Org/MiniMax-H3` 的单文件没有 revision pin**,身份判据只有字节数 | 上游无声重传则发现不了 |
| 3 | 5090 与 runpods 的 **L2 从未跑过** | 磁盘布局的正确性在这两台机器上只是「继承自 6000a」 |
| 4 | 6000a 的 **HF cache 桥接声明 `b3c7290e…`,磁盘实际是 `101ecd0a…`** | 溯源时会指向错误的 revision;复现请用后者 |
| 5 | 5090 的 **#1 来源证据打架**(rsync vs `hf download`) | 对复现无影响,但溯源结论只能标「不确定」 |
| 6 | `merge_script_sha256` 记的脚本版本已不存在、不可复现 | 已永久排除出门禁;二选一的修法见各机档案第 8 节 |
| 7 | 三机合计约 **117G 可回收残留**(5090 ~50G / 6000a 11G / runpods 56G) | 6000a 两块盘 93–94% 满,最需要清 |
