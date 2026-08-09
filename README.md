# H3_inference — MiniMax-H3 首尾帧视频生成的多机推理产线

把 [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 的 FL2VA（首帧+尾帧→带
原生音频的视频）跑成可复现的生产产线：一套 CLI 从一台工作机编排多台 GPU 机器，
支持 ComfyUI / vLLM-Omni / SGLang 三种后端与 Turbo-LoRA 少步蒸馏。

**本仓库只含代码、配置与实验档案，不含模型权重**（权重合计数百 GB，各机自行获取，
见下）。

## 现在能跑多快

864×480 / 124 帧（≈5.17s）/ 带音频，warm 延迟：

| 机器 | 最快产线 | NFE | 延迟 | 占卡 |
|---|---|---|---|---|
| 4× RTX 5090（云） | `vllm-fp8-turbo-tp2u2` | 4 / 6 | **16.3s / 18.5s** | 4 |
| 4× RTX 6000 Ada | `vllm-fp8-turbo-tp4` | 4 / 6 | 17.9s / 23.5s | 4 |
| 2× RTX 5090 | `vllm-fp8-turbo-tp2` | 4 / 6 | 22.0s / 29.0s | 2 |
| 2× RTX 5090 | `comfy-int8-turbo-1c` | 4 / 6 | 25.0s / 30.0s | **1** |

对照：未加速的 ComfyUI 单卡 30 步基线约 100s。完整对比见 `doc/*_results.md`。

## profile 命名

`<引擎>-<精度>-<变体…>-<拓扑>`

- **精度** = DiT 权重精度：`int8` / `fp8` / `bf16` / `nvfp4`
- **变体** = 会改变输出或步数的技术选择，可叠加：`original`（零 tweak）·
  `teacache` / `sage` / `flash`（显式加速方法）· `turbo`（Turbo LoRA 少步蒸馏）
- **拓扑** = ComfyUI 用计算卡数 `1c`/`4c`；引擎用并行度 `tp2`/`tp4`/`tp2u2`

不影响输出的东西（内存/调度 flag、引擎的 attention backend）**不进名字**，它们是参数。

```bash
scripts/h3_generate.py --list          # host × profile 能力矩阵与默认步数
```

## 快速上手

前提：工作机只需 `ffmpeg` / `ssh` / `scp`（**零 pip 依赖**，纯标准库），
且 `~/.ssh/config` 里配好目标机别名。

```bash
# 生成一条(自动切后端、等就绪、回传 mp4,并分别报告启动与推理耗时)
scripts/h3_generate.py --first a.png --last b.png --prompt-file p.txt \
    --host 6000a --profile vllm-fp8-turbo --nfe 6

# 把一台机器上所有 profile 用同一素材跑一遍,出对比表
scripts/h3_eval.py --first a.png --last b.png --prompt-file p.txt \
    --host runpods --timed 3
```

### 步数一律用 NFE

`--nfe` = **实际 DiT forward 次数**。这不是多余的抽象：vLLM/SGLang 的
`num_inference_steps=N` 只跑 **N−1** 次 forward（N 个 sigma 点），而 ComfyUI 是
`steps == forwards`。客户端按各自语义换算，`scripts/test_h3_schedule.py` 会验证
换算后的 sigma 网格与 LoRA 作者的解析式逐点一致（≤1e-7）。

## 仓库即项目根

每台机器上，本仓库**就是**项目根（`~/data/dropbox/CV/h3` 或 `/workspace/h3`），
`.gitignore` 把权重、第三方源码、产物挡在外面。所以 `git pull` 就是更新代码，
新机器 `git clone` 完目录结构直接对。

```
scripts/        全部代码(编排 CLI + 各机产线脚本 + 工具与测试)
doc/            方案分析与 benchmark 全表
doc/machines/   三台 GPU 机的实测档案 + 跨机对照 + 权重台账 + pip/conda 锁定清单
claude_history/ 实验档案(每个主题一份 FINAL.md 权威结论)
gallery/        对比展示页(媒体文件不入库)
workflows/stage_a/   基准提示词 7 则(ToS 场景),归档 benchmark 的原始输入
inputs/         运行时素材目录(图片/视频不入库)
```

被忽略的（各机自行获取/生成）：`models*/ base/ merged/ loras/ src*/ ComfyUI/ env/
outputs/ logs/ run/`。

## 装机

**先读 [`doc/machines/`](doc/machines/)** —— 三台 GPU 机的实测档案（硬件/驱动/env/第三方源码
与补丁/权重落盘/运行期环境变量/从零复现步骤），外加跨机对照与「新机器该装什么」的决策树：

| 想干什么 | 看哪份 |
|---|---|
| 拿到一台新机器，不知道装哪套 wheel、能不能跑 serving | [`doc/machines/README.md`](doc/machines/README.md) §3 决策树 |
| 三台机器哪里一样、哪里不一样 | [`doc/machines/README.md`](doc/machines/README.md) §1–§2 |
| 在某台机器上复现 / 排障 / 重装 | [`popos-5090.md`](doc/machines/popos-5090.md) · [`popos-6000a.md`](doc/machines/popos-6000a.md) · [`runpods-5090x4.md`](doc/machines/runpods-5090x4.md) 第 7 节 |
| 下权重 / 权重溯源 / 算磁盘预算 | [`doc/machines/models.md`](doc/machines/models.md) |
| 精确到包版本 | `doc/machines/locks/`（每机每环境的 `pip freeze` + `conda env export`） |
| **为什么**这么做（实验证据链） | `claude_history/06_infra/FINAL.md` 及各主题 `FINAL.md` |

装机脚本在 `scripts/`：

- `setup_h3.sh` / `install_backends*.sh`（本地 GPU 机）
- `setup_runpods.sh` / `post_setup*.sh` / `fix_cu129.sh`（云机）
- `download_h3.sh` / `download_bf16.sh`（权重）
- `merge_turbo_lora.py`（把 Turbo LoRA 合进官方 BF16，产出 `.complete` 契约）

⚠️ **`pr5910_resident_stride_fix.patch` 必须打上**。不打的话 FP8 + DLO 路径不会报错，
而是**静默产出纯噪声**。判断补丁在不在要看代码内容，但**必须限定在
`PinnedResidentLayerGroup` 类内**：

```bash
sed -n '/class PinnedResidentLayerGroup/,/def offload/p' \
    <src>/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py | grep -q as_strided
```

裸的 `grep as_strided <整个文件>` **恒真、毫无鉴别力**——未打补丁的原文件在 streamed 路径
（299/412 行）本来就有两处；`h3_switch_*.sh` 与 `setup_runpods.sh` 用的就是这个坏判据，
日志抬头的 `stride_patch=APPLIED_as_strided` 因此**不能当证据**（详见
[`doc/machines/README.md`](doc/machines/README.md) §2.3）。也不能用 `git diff`
（补丁已提交时 worktree 干净，会反过来误判为"没打"）。

## 安全

- **不要把任何密钥放进本仓库**。`credentials/` 已被 `.gitignore` 排除；
  HF token 走 `~/.cache/huggingface/token` 或 `HF_HOME`。
- 基准素材统一用 [Tears of Steel](https://mango.blender.org/)（CC-BY）。
  本项目不使用也不分发未授权的真实可识别人物影像，详见 `claude_history/CLAUDE.md`。

## 路线（Phase 2）

1. `machines.yaml` 外置机器台账（路径根/端口/GPU/env/profile 白名单），
   现在这份台账还在 `scripts/h3_generate.py` 的 `HOSTS` 字典里
2. 三份 `h3_switch*.sh`（446 行、三套互不兼容的 CLI 契约）合成一份参数化版
3. 5 个 `launch_comfy*.sh` 合 1
4. `bootstrap.sh` 一键装机 + 机器自识别（驱动→wheel 变体、拓扑→TP 候选、
   RAM→serving 可行性）+ 健康自检出片

## 许可

MIT，见 `LICENSE`。模型权重与上游依赖各自遵循其原始许可。
