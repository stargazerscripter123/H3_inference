# FastH3 Omni环境与构建记录

记录日期：2026-10-21，Australia/Sydney。目标机器：`popos-5090`，双RTX5090。

本记录包含本次实际完成的环境、原生kernel和基础设施验证，以及尚未完成的端到端验收。其来源是主会话的远端回执；下述步骤是已执行工作的整理，不是逐字安装命令转录。完整freeze和构建身份见 `environment/environment-final.freeze.txt`、`environment/environment-provenance.json`；模型预期身份见 `environment/expected-assets.json`。后者是下载校验依据，不代表所有文件已下载完成。

## 源码、环境和路径

| 项目 | 固定版本或当前路径 |
| --- | --- |
| 项目根 | `/home/isaac/workdir/h3` |
| 项目部署代码 | commit `9443463`，已push且远端已pull |
| FastVideo源码 | `/home/isaac/workdir/h3/src/FastVideo` |
| FastVideo commit | `cdda1b7535fe32699335e39db82c1d4db21ceb88` |
| Python | managed Python `3.12.15`，venv为`/home/isaac/workdir/h3/envs/omni` |
| Python入口 | `/home/isaac/workdir/h3/envs/omni/bin/python` |
| uv | `0.13.4`；`/home/isaac/workdir/h3/.bootstrap/bin/uv` |
| Torch | `2.12.0`，cu130 |
| Ray | `2.59` |
| NVCC | NVIDIA `13.0.88` |
| CCCL | NVIDIA `13.0.85` |
| CUDA_HOME | `/home/isaac/workdir/h3/envs/omni/lib/python3.12/site-packages/nvidia/cu13` |
| CUDACXX | `/home/isaac/workdir/h3/envs/omni/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc` |
| 媒体工具目录 | `/home/isaac/workdir/h3/.tools/media/bin` |
| ffmpeg | `9.0.2`；ffprobe由上述同一目录提供 |

宿主系统为Pop!_OS24.04、驱动595.104.02。当前环境建立在项目目录中，使用managed Python提供的开发头文件。旧系统Python3.12.3和旧conda配方不是最终构建环境。

保留venv的Python入口，不把其symlink解析为外部managed Python的实体路径后再启动；否则可能失去venv的包定位。CUDA默认路径也必须从venv目录推导。

## 本次完成的构建步骤

1. 在专用venv中安装managed Python3.12.15所需环境和Torch2.12/cu130，保留固定FastVideo源码。
2. 在该venv的`site-packages/nvidia/cu13`下组成NVIDIA pip CUDA13.0工具链，使用NVCC13.0.88与CCCL13.0.85。
3. 补齐该环境内的`lib64 -> lib`和链接器需要的无版本`.so`链接。修改范围为项目venv；不把旧环境的路径或系统级CUDA目录当作这次构建的工具链。
4. 从固定FastVideo源码中的`fastvideo-kernel`构建并安装目标为`sm_120a`的wheel。实际构建选项如下。
5. 使用当前环境运行下节记录的原生GPU、NCCL、Ray及配置检查，分别保留通过范围。

| CMake选项 | 本次值 | 作用范围 |
| --- | --- | --- |
| `FASTVIDEO_KERNEL_BUILD_TK` | `OFF` | 不构建ThunderKittens分支 |
| `FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER` | `ON` | 构建原生FP4 attention/quantization分支 |
| `FASTVIDEO_KERNEL_BUILD_ULYSSES_A2A` | `OFF` | 不构建可选NCCL-device Ulysses扩展 |
| 编译目标 | `sm_120a` | RTX5090原生kernel目标 |

这些完整选项名已与固定源码的`fastvideo-kernel/CMakeLists.txt`核对。`ULYSSES_A2A=OFF`仅指可选编译扩展，不表示禁用标准NCCL集合通信；本次双卡NCCL探针已通过。构建成功也不能替代完整Omni执行测试。

复现时须保持同一源码、Python、Torch/CUDA ABI及上述选项，并记录实际安装命令、构建日志和产出wheel的hash。仅记录包的发行版本号，不能识别它是否由这份源码、这组架构选项构建。本记录暂不提供未经回执核对的“一键重装”命令。

## 独立启动时的工具链环境

部署包`omni_control.py`在`validate`和`start`子进程中自动设置：

- `CUDA_HOME`：默认`<venv>/lib/python3.12/site-packages/nvidia/cu13`。
- `CUDACXX`：`<CUDA_HOME>/bin/nvcc`。
- `PATH`：依次前置venv的`bin`、CUDA的`bin`、`<项目根>/.tools/media/bin`，后接调用者的`PATH`。

可用绝对路径环境变量`OMNI_CUDA_HOME`、`OMNI_MEDIA_BIN`覆盖。默认布局对应本次Python3.12环境。这样从新shell独立启动服务仍能找到JIT编译器和ffmpeg/ffprobe，无需依赖安装时shell的手工export。

直接执行的AdaLN准备脚本、HTTP客户端和benchmark不经过控制器，需要它们自己的进程环境能找到相应工具；部署包README给出了对应环境设置。服务配置指纹包含这组环境，变更后需显式stop/start。

## 实际验证记录

| 验证层 | 已确认结果 | 验收界限 |
| --- | --- | --- |
| 原生sparse attention | 2项GPU测试通过 | 证明受测原生kernel路径能执行，不代表完整模型或任意shape |
| NCCL双卡通信 | all-to-all与round-trip探针通过 | 不代表Omni SP2已出片，也不是性能数字 |
| Ray健康查询接口 | Ray2.59私有GCS查询能读取指定actor的`ALIVE`状态 | 验证health adapter依赖的接口；未据此宣布完整服务健康 |
| 配置到引擎转换 | SP1与SP2实际`ServeConfig`及转换均通过 | 不含完整模型实例化和真实生成 |
| 部署包CPU检查 | 31项测试通过 | 与上列GPU测试分开计数 |
| 代码分发 | `5734dd1`已push/pull | 指该版本已到远端；后续改动须另记录实际commit |

源码wheel位于目标机 `wheels/fastvideo-cdda1b75-cu130-sm120a/fastvideo_kernel-0.3.5-cp312-cp312-linux_x86_64.whl`，SHA256为 `e3b7537f3aeb46e2986e5501f895d62dc3d763123bca4d6a5857615eb7526dc0`。构建日志：`logs/omni/kernel-build-03.log`。

GPU补充验证：FlashInfer/VAE测试5 passed（含四种fused/eager逐位一致组合），日志 `logs/omni/flashinfer-vae-tests-01.log`；FFN NVFP4与Transformers5.15视觉接口8 passed，日志 `logs/omni/ffn-vision-tests-01.log`。完整pipeline导入及本地processor处理全零合成图像/视频接口通过；没有读取图片文件。以上测试都不代替端到端生成。

## 资产与生成验收仍待完成

目标QAD revision为`f01e6a23ac6066aa9eb622cf94b6259442e70ec2`。当前47个目标文件、约63.87GiB的下载仍在进行；完整组件、实际hash和最终manifest尚不能据本记录判定为通过。36层packed decoder、采样契约和AdaLN表要分别记身份。

以下项目尚未验证：

- 下载完成后的全部必要文件与资产manifest校验。
- AdaLN准备及新进程table-only执行，含有/无显式参考音频两类clock。
- SP1完整生成、124/362帧及参考图/音频/视频场景。
- SP2完整生成、双rank实际参与和服务完整启动/停止流程。
- ffprobe真实产物检查，以及用户对落盘画面与音频的质量验收。
- 每卡峰值显存、主机内存、稳定延迟、SP1/SP2加速比和15秒目标。

后续结果按机器、源码/资产版本、SP配置、8次PDD forward、输入场景、冷热态和缓存条件分别追加。31项CPU测试、2项原生GPU测试、通信成功与配置转换成功不合并成“Omni部署已完成”。
