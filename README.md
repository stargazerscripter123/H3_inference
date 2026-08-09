# H3 ComfyUI · 2×RTX 5090 单机部署

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 的 **FL2VA**（首帧 + 尾帧 →
带原生音频的视频），在一台 2×RTX 5090 的机器上跑成产线。三个 profile，一条本地命令出片。

**本仓库不含模型权重**（最小集 40.3 GiB，`install/download_models.sh` 获取），
也不含 ComfyUI 本体与 custom_nodes（`install/bootstrap.sh` 按钉死的 commit 拉取）。

## 装

```bash
git clone -b comfy-5090-minimal https://github.com/stargazerscripter123/H3_inference.git h3
cd h3 && bash install/bootstrap.sh        # 环境 + ComfyUI@pin + 3 个节点@pin + 40.3G 权重
bash install/doctor.sh                    # 自检,应全绿
```

`bootstrap.sh --dry-run` 可以先预演每一步。跑一次要几十分钟，绝大部分是下权重。

## 跑

```bash
./scripts/h3.sh --profile comfy-int8-turbo-1c --nfe 4 \
    --first first.png --last last.png --prompt-file prompts/smoke.txt
```

产物落 `outputs/<name>/`，同目录一份 `*.manifest.json` 记录这次的全部参数与环境指纹。
`./scripts/h3.sh --list` 看 profile，`--help` 看全部参数。

## 三个 profile

864×480 / 124 帧（5.17s）/ 带原生音频，本机实测 **warm** 延迟：

| profile | 采样 | 步数 | GPU:端口 | warm | 冷启首推 |
|---|---|---|---|---|---|
| `comfy-int8-turbo-1c` | Turbo LoRA 少步蒸馏 | 4 | GPU0:8190 | **25.0s** | 35.1s |
| `comfy-int8-teacache-1c` | TeaCache 有损跳步 | 12 | GPU0:8189 | **35.0s** | 40.0s |
| `comfy-int8-original-1c` | 零 tweak 基线 | 30 | GPU1:8188 | **110.0s** | 125.0s |

三者共用同一套权重（int8 DiT + nvfp4 TE + video/audio VAE），只有采样路径不同。

两条会咬人的规矩：

- **新起的 worker，第一次推理是冷的。** ComfyUI 的 `/system_stats` 在模型加载之前就
  响应（6s 就"就绪"），35G 权重是首次提交时才懒加载的。要量速度就换 `--seed` 再跑一遍。
- **量速度必须换 seed。** ComfyUI 缓存整张图，参数完全相同的第二次提交直接返回缓存。

## 文件地图

```
scripts/h3.sh            ★ 入口:选 profile → 起 worker → 统一画布 → 生成 → 报时
scripts/run_fl2va.py     ★ 生成本体:造 ComfyUI 图 → POST /prompt → 轮询 → 取回 mp4
scripts/comfyctl.sh        worker 启停与互斥(RAM 125G 装不下三个 worker,这是必需的)
scripts/prep_frames.sh     ffmpeg 统一画布(scale-to-cover + center-crop)
scripts/extract_frames.sh  可选:从源视频抽首尾帧对

install/pins.env         ★ 唯一的版本真相:ComfyUI/节点 commit、HF revision、
                           权重 bytes + sha256。每个 pin 都写了"为什么是它"
install/bootstrap.sh       装机(幂等,支持 --dry-run / --only <step>)
install/download_models.sh 权重下载 + 校验(支持 --verify-only)
install/doctor.sh          自检,出结论
install/requirements.lock.txt  权威包列表(108 个包,自带 PyTorch cu130 index)
install/environment.yml    conda 侧快照,**不用来建环境**,只用来 diff
```

细节都在各文件的头部注释里，这里不重复。

## 硬件前提

实测基准机：Ubuntu 24.04.4 / 驱动 580.173.02（CUDA 13.0）/ 2×RTX 5090 32G /
RAM 125 GiB / 磁盘 ≥60 GiB 可用 / ffmpeg。

驱动低于 580 跑不了 `torch 2.11.0+cu130`。RAM 是最硬的约束——**单个 ComfyUI worker
常驻 46.4 GiB**，所以 `comfyctl.sh` 强制互斥：original(GPU1) 与 teacache(GPU0) 可共存，
turbo 独占。

## 几条不报错的坑（详见对应文件注释）

- **Turbo LoRA 必须走 `MiniMaxH3TurboLoRA` 节点。** ComfyUI 核心的
  `LoraLoaderModelOnly` 对 H3 是静默 no-op——匹配 0/518 个张量**且不报错**。
  验收标志是 worker 日志出现 `208 backbone modules … 51 adaln injected`。
- **权重只校验字节数挡不住下错文件。** 同一 HF repo 里 `ref2va` 与 `fl2va` 系列
  字节数完全相同，所以 `pins.env` 里钉的是 sha256。
- **ComfyUI 版本不能随便升。** 钉的这版还没有原生 `ModelSamplingAV`，Turbo 节点会回落到
  自带的 legacy dual-schedule；升级会切到原生实现，**输出会变**。

## 与上游主仓库的关系

本分支是 [H3_inference](https://github.com/stargazerscripter123/H3_inference) 的
**orphan 分支**，与 `main` 没有共同祖先，不能 merge / cherry-pick。`main` 是三机
（+vLLM-Omni / SGLang serving）的完整产线与实验档案；这里只留这一台机器的 ComfyUI 部分。

## 许可

MIT，见 `LICENSE`。模型权重与上游依赖各自遵循其原始许可。
