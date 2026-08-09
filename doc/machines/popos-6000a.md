# popos-6000a — 机器档案

> 采集日期 **2026-08-09**(远端时钟 21:31–21:45 AEST)。绝大多数事实来自本次 ssh 实测命令输出；
> 少数标注了出处的事实来自 HF 公开 API(§7 步骤 6 的 repo `gated`/文件列表)与 Mac 仓库内的脚本源码；
> 凡推测都显式标注。与 `claude_history/` 历史档案冲突处，**以实测为准**并在正文点明。
> 本次采集**严格只读**：除把清单写进 Mac 仓库 `doc/machines/locks/` 外，未改远端任何文件。

---

## 1. 这台机器是什么

**四卡 RTX 6000 Ada 的"全能机"**：三条 serving 产线(vLLM-Omni / SGLang / ComfyUI)都在这台机器上跑通，
是项目里唯一同时具备 **BF16 4 卡 oracle**(DisTorch2)和 **TP4 引擎 serving** 能力的节点。
也是官方 145G BF16 权重与 Turbo merged checkpoint 的**源头机**(5090 与 runpods 的权重都从这里派生或复算)。

| 项 | 实测值 |
|---|---|
| 主机名 | `popos-6000a`(ssh 别名同名) |
| GPU | 4 × **NVIDIA RTX 6000 Ada Generation**，每卡 **49140 MiB**(48 GiB)，compute capability **8.9**(sm_89)，功耗上限 300 W |
| GPU PCI | `16:00.0` / `34:00.0` / `AC:00.0` / `CA:00.0` |
| 互联 | `nvidia-smi topo -m` 四卡两两 **NODE**(同 NUMA 内跨 PCIe host bridge)，无 NVLink |
| **P2P** | `topo -p2p r` 与 `-p2p w` **四卡两两全 OK** —— 与 runpods 的全 `CNS`、5090 的无 P2P 形成对比，这是本机能用 DisTorch2 跨卡与 TP4 的物理前提 |
| CPU | Intel **Xeon w9-3475X**，36 核 / 72 线程，单 socket，**NUMA 节点只有 1 个**(0-71 全在 node0)，L3 82.5 MiB，max 4.8 GHz |
| 内存 | **376 GiB**(采样时 used 35 / free 150 / buff-cache 190 / available 317) |
| Swap | 19 GiB(`zram0` 16G + `cryptswap` 4G) |
| `/dev/shm` | 189 GiB |
| 磁盘 | `/` = LUKS+LVM on nvme0n1p3，**3.7T 已用 3.3T(94%)，余 244G**；`/home/isaac/Data` = LUKS on nvme1n1，**3.7T 已用 3.3T(93%)，余 284G**；`/home/isaac/Backup` = LUKS on sda 9.1T(26%) |
| 项目根 | `/home/isaac/data/dropbox/CV/h3`(= git 工作区，remote `stargazerscripter123/H3_inference`，HEAD `39c4b5089f78f25ae62b5df19fa2e5f9492768c8`，worktree 干净，与 Mac 仓库同 HEAD) |
| 权重区 | `/home/isaac/Data/h3_weights`(第二块 NVMe，避开 94% 满的根盘) |
| 采集时在跑 | vLLM-Omni `:8091` 服 `MiniMax-H3-Turbo-v4s600ema/FL2VA` FP8 TP4(`run/vllm.variant = turbo-fp8`)+ ComfyUI `:8288` 常驻；四卡各占 **37380 / 37576 / 38686 / 37562 MiB**(36.5–37.8 GiB) |

### 它承担哪些 profile

来自 `scripts/h3_generate.py --list` 的 6000a 那几行(全部 ✓ 都在本机)：

| profile | 后端 | 默认步数 | 画布 |
|---|---|---|---|
| `comfy-int8-original-1c` | comfy:gpu0,1,2,3:8288 | 30 步 | 832×480 |
| `comfy-bf16-original-4c` | comfy:DisTorch 4卡 | 30 步 | 832×480 |
| `vllm-bf16-original-tp4` | serving:`vllm bf16` | NFE 11 | 864×480 |
| `vllm-fp8-original-tp4` | serving:`vllm fp8` | NFE 11 | 864×480 |
| `vllm-bf16-turbo-tp4` | serving:`vllm-turbo bf16` | NFE 6 | 864×480 |
| `vllm-fp8-turbo-tp4` | serving:`vllm-turbo fp8` | NFE 6 | 864×480 |
| `sglang-fp8-original-tp4` | serving:`sglang` | NFE 11 | **1376×768**(API 硬锁短边 768) |
| `sglang-fp8-turbo-tp4` | serving:`sglang-turbo` | NFE 6 | **1376×768** |

另有 `h3_switch.sh sglang-lora`(dynamic LoRA，研究/AB 用)不在 profile 矩阵里。
三条 serving 产线**互斥占 4 卡**，由 `scripts/h3_switch.sh` 串行化管理；ComfyUI 进程常驻但切换时用 `POST /free` 卸模型。

---

## 2. 系统与驱动

| 项 | 实测值 | 出处 |
|---|---|---|
| 发行版 | **Pop!_OS 22.04 LTS**(jammy，`ID_LIKE=ubuntu debian`) | `/etc/os-release` |
| 内核 | `7.0.11-76070011-generic`(PREEMPT_DYNAMIC) | `uname -a` |
| NVIDIA 驱动 | **580.173.02** | `nvidia-smi` |
| 驱动自报 CUDA | **13.0** | `nvidia-smi` 表头 |
| gcc / g++ | `/usr/bin/gcc` **11.4.0**(Ubuntu 11.4.0-1ubuntu1~22.04.3) | `gcc --version` |
| 系统 CUDA toolkit | **没有**(`which nvcc` 空，`/usr/local/cuda*` 不存在) | 实测 |
| make / cmake | `/usr/bin/make`、`/usr/bin/cmake` 存在 | 实测 |
| ninja | 来自 conda 环境，`1.13.0.git.kitware.jobserver-pipe-1` | `envs/h3_sglang_NV_py312/bin/ninja --version` |
| conda | `26.5.3`，安装在 `/home/isaac/miniconda3`，channels 只有 `defaults` | `conda --version` / `conda config --show channels` |

> **档案缺口已补**：历史考古报告第五节写「6000a 驱动没记」。实测 **580.173.02**，
> 与档案里 5090 的 580.173.02 相同。

### 驱动版本决定了哪些 wheel 变体

- CUDA 13.0 运行时要求驱动 **≥ 580.65**；580.173.02 满足 → **本机可以直接用 `+cu130` wheel**。
- 因此本机 torch 全部是 **2.11.0 / CUDA 13.0**，**不需要** runpods 那套 `+cu129` 绕行方案
  (runpods 驱动 570.195.03 只到 CUDA 12.8，必须换 `vllm-0.26.0+cu129` wheel)。
  **别把 runpods 的 `fix_cu129.sh` 套到这台机器上。**
- 没有系统 CUDA toolkit → 所有需要 `nvcc` / `libcudart` 的 JIT(SGLang)必须靠
  pip 包 `nvidia-cuda-nvcc` + `nvidia-cuda-runtime` 提供，`CUDA_HOME` 指到
  `site-packages/nvidia/cu13`。这就是 `h3_switch.sh` 里那一串 `CUDA_HOME/PATH/LIBRARY_PATH/LD_LIBRARY_PATH` 的由来(见第 6 节)。

---

## 3. Python 环境

`conda env list` 共 18 个环境，**与 H3 相关的只有 3 个**(其余是 krea/rfdetr/tts 等他人或旧项目)。
项目根下**没有** venv(`env/`、`venv/`、`.venv/` 均不存在)——与 runpods 的 venv 布局根本不同。

| conda env | 路径 | Python | pip | torch | 服务于哪条产线 |
|---|---|---|---|---|---|
| `h3_comfy_NV_py312` | `/home/isaac/miniconda3/envs/h3_comfy_NV_py312` | 3.12.13 | 26.1.2 | 2.11.0+cu130 | **ComfyUI :8288**(`launch_comfy_6000a_4gpu.sh` 的 `ENVPY`)；**同时是 `h3_generate.py` 的默认 `remote_python`**，所有 profile 的客户端 `run_fl2va.py` / `run_fl2va_vllm.py` / `run_fl2va_sglang.py` 都在这个 env 里跑；`dl_fl2va_auth.sh` 也从这里取 `hf` CLI |
| `h3_vllm_NV_py312` | `…/envs/h3_vllm_NV_py312` | 3.12.13 | 26.1.2 | 2.11.0+cu130 | **vLLM-Omni :8091**(`h3_switch.sh` 的 `VLM_BIN`) |
| `h3_sglang_NV_py312` | `…/envs/h3_sglang_NV_py312` | 3.12.13 | 26.1.2 | 2.11.0+cu130(pip 元数据记为 `2.11.0`) | **SGLang :30010**(`h3_switch.sh` 的 `SGL_BIN`，`CUDA_HOME` 也指向本 env 的 `nvidia/cu13`) |

三个环境 `torch.cuda.device_count()` 都是 4，`torch.version.cuda` 都是 `13.0`，cudnn `91900`。

> ⚠️ **`h3_sglang` 的 torch 只有 pip 元数据丢了 `+cu130`,运行时版本串是一样的**。实测:
> 三个 env 的 `torch.__version__` **都是 `2.11.0+cu130`**;但 sglang env 的
> `pip show torch` / `pip freeze` 记为 `2.11.0`,dist-info 目录名是 `torch-2.11.0.dist-info`
> (vllm env 是 `torch-2.11.0+cu130.dist-info`)。两个 env 的 `torch.version.git_version` 都是
> `70d99e998b4955e0049d13a98d77ae1b14db1f45` —— **同一个 cu130 wheel**,只是 uv 安装时丢了
> local version 标记(两个 dist-info 的 `INSTALLER` 都是 `uv`,所以也不是"uv vs pip"的区别)。
> 复现时**不要**因为 `pip freeze` 少了 `+cu130` 就去重装 torch。

两个**看名字像但与 H3 无关**的环境(实测已排除)：`sgldiff_NV_py312`(sglang 0.5.13 wheel)、
`vllm_omni_NV_py312`(vllm 0.23.0)——都建于 2026-06，早于本项目(2026-08-05 起)，
且没有任何 H3 脚本引用它们。**不要误装到这两个里面。**

### 3.1 `h3_comfy_NV_py312`(ComfyUI + 全部客户端)

| 包 | 版本 | 备注 |
|---|---|---|
| torch / torchvision / torchaudio | **2.11.0+cu130** / **0.26.0+cu130** / **2.11.0+cu130** | `--index-url …/whl/cu130` |
| triton | 3.6.0 | |
| numpy | **2.4.4** | 注意与另外两个 env 的 2.3.5 不同 |
| transformers | 5.14.1 | |
| safetensors | 0.8.0 | |
| **comfy-kitchen** | **0.2.26** | 量化算子，ComfyUI `requirements.txt` 内钉死，**必装** |
| **comfy-aimdo** | **0.4.11** | 同上 |
| comfyui-frontend-package | 1.47.11 | 同样在 requirements 里钉死 |
| huggingface_hub | 1.26.0 | 带 `hf-xet 1.6.0`；另装了 `hf_transfer 0.1.9`(已废弃，见第 8 节) |
| nvidia-cuda-runtime / cudnn-cu13 / nccl-cu13 | 13.0.96 / 9.19.0.56 / 2.28.9 | `nvidia/cu13/lib` 就是 launcher 里 `LD_LIBRARY_PATH` 指的目录 |

**没有** flash-attn / sageattention / xformers / nvcc —— ComfyUI 侧走 torch SDPA，不需要它们。
全部 105 行见 `doc/machines/locks/popos-6000a.h3_comfy_NV_py312.pip-freeze.txt`。

### 3.2 `h3_vllm_NV_py312`(vLLM-Omni)

| 包 | 版本 | 安装形态 |
|---|---|---|
| **vllm** | **0.26.0** | PyPI wheel，`Location: …/site-packages` |
| **vllm-omni** | **`0.1.dev1+ga874b8e09`** | **editable，`Editable project location: /home/isaac/data/dropbox/CV/h3/src/vllm-omni`**；`pip freeze` 记为 `-e git+https://github.com/vllm-project/vllm-omni.git@a874b8e09bb5251bb16e3de8bc22edf3880eec09#egg=vllm_omni` |
| torch / torchvision / torchaudio | 2.11.0+cu130 / 0.26.0+cu130 / 2.11.0+cu130 | |
| transformers / diffusers / accelerate | 5.14.1 / **0.38.0** / 1.12.0 | |
| flashinfer-python | 0.6.14 | |
| triton / numpy / safetensors | 3.6.0 / 2.3.5 / 0.8.0 | |
| nvidia-cuda-nvcc / cuda-runtime | 13.3.73 / 13.0.96 | vllm 依赖链带进来的 |
| humming-kernels / quack-kernels / tilelang / nvidia-cutlass-dsl | 0.1.10 / 0.6.1 / 0.1.9 / 4.6.0 | vllm 0.26.0 的量化 kernel 依赖 |

**没有** flash-attn / sageattention / xformers。`--diffusion-attention-backend FLASH_ATTN` 用的是
vLLM 自带的 FA 实现，不是 pip 的 `flash-attn` 包(实测该包在本 env 不存在)。

> ⚠️ `import vllm_omni` 会打一条 **RuntimeWarning: vLLM and vLLM-Omni appear to have mismatched
> major/minor versions(0.1.dev1+ga874b8e09 vs 0.26.0)**。这是 setuptools_scm 从 git 描述算出的
> 版本号与 vllm 主版本对不上导致的**纯提示**，产线正常工作。别为了消掉它去改版本。

全部 227 行见 `locks/popos-6000a.h3_vllm_NV_py312.pip-freeze.txt`。

### 3.3 `h3_sglang_NV_py312`(SGLang)

| 包 | 版本 | 安装形态 |
|---|---|---|
| **sglang** | **`0.0.0.dev1+g407a65d3c`**(元数据) | **editable，`Editable project location: /home/isaac/data/dropbox/CV/h3/src/sglang/python`**；`pip freeze` 记为 `-e git+https://github.com/sgl-project/sglang.git@4c28e247ecd8ffa9b4507a90c1cf854bb734cc8f#egg=sglang&subdirectory=python` |
| sglang-kernel / sgl-deep-gemm | 0.4.5 / 0.1.5.post1 | |
| torch / torchvision / torchaudio | 2.11.0+cu130 / 0.26.0+cu130 / 2.11.0+cu130 | `pip freeze` 记为 `2.11.0` / `0.26.0` / `2.11.0`(uv 装时丢了 local version);`torch.__version__` 实测带 `+cu130`,与另两个 env 同 wheel |
| transformers / diffusers | **5.12.1** / **0.37.0** | 比 vllm env 低一档，两个 env 互不干扰 |
| flash-attn-4 | **4.0.0b19** | 只有这个 env 有 |
| flashinfer-python | 0.6.15.post1 | |
| **nvidia-cuda-nvcc** / **nvidia-cuda-runtime** | **13.3.73** / **13.0.96** | SGLang JIT 编译必须；系统无 nvcc |
| runai-model-streamer | 0.16.1 | 对应 `RUNAI_STREAMER_MEMORY_LIMIT` 环境变量 |
| modelscope / modelscope-hub | 1.39.1 / 0.2.0 | HF 限速时的换源手段 |
| uv | 0.12.1 | 安装用 |

**JIT 依赖的实体已就位(实测)**：
- `…/site-packages/nvidia/cu13/bin/nvcc` 存在
- `…/site-packages/nvidia/cu13/lib/libcudart.so` 是指向 `libcudart.so.13` 的**无版本 symlink**
  (`-lcudart` 链接必须靠它)

> ⚠️ **`pip show sglang` 报的版本 `g407a65d3c` 是安装时快照，已经过期**。当前 checkout 的实际 HEAD 是
> `4c28e24`(turbo-lora-backport 分支)。editable 安装下代码以工作区为准，**换分支不需要重装**，
> 但**判断装了什么必须看 git，不要看 `pip show`**。`pip freeze` 报的 commit 才是当前的。

全部 236 行见 `locks/popos-6000a.h3_sglang_NV_py312.pip-freeze.txt`。

---

## 4. 第三方源码与补丁

### 4.1 checkout 表(实测 `git remote/branch/rev-parse/status`)

> **日期口径 = committer date(`git log -1 --format=%cd --date=short`)**,不是 author date。
> 两者会差(例如 sglang `4c28e24` author=2026-08-07 / commit=2026-08-08;
> ComfyUI `57500fc` author=2026-08-03 / commit=**2026-08-02**,author 反而更晚)。溯源时按 `%cd` 对。

| 路径 | remote | branch | HEAD(完整) | 短 | dirty | 提交日期(`%cd`) / 标题 |
|---|---|---|---|---|---|---|
| `~/data/dropbox/CV/h3/ComfyUI` | `comfyanonymous/ComfyUI` | detached `FETCH_HEAD` | `57500fc5bc92566a63f2046824f522cd55c335ca` | `57500fc` | 0 | 2026-08-02 · `feat: Support MiniMax-H3 (CORE-375) (#15224)`；`comfyui_version.py` = **0.29.0** |
| `~/data/dropbox/CV/h3/src/vllm-omni` | `vllm-project/vllm-omni` | `main` | `a874b8e09bb5251bb16e3de8bc22edf3880eec09` | `a874b8e` | 0 | 2026-08-07 · `[Perf][CI] Add MiniMax-H3 4xH100 diffusion perf config (#5836)` |
| `~/data/dropbox/CV/h3/src/vllm-omni-pr5910` | 同上 | **`pr5910-stride-fix`** | `070096bd6872418cd8c4e1be18ba1450f301180e` | `070096bd` | 0 | 2026-08-09 · `fix(dlo): keep FP8 transposed-view stride when repointing resident layers`；`git describe` = `v0.26.0-55-g070096bd`，父提交 = `b18eeff2`(PR#5910 的 `[Diffusion][Quantization] Enable MiniMax-H3 global FP8 with DLO`)；本地另有 ref `pr5910-head = 1a9b9c2c13c97763567405f43c5fee994c43ab13`(runpods 用的那个 head) |
| `~/data/dropbox/CV/h3/src/sglang` | `sgl-project/sglang` | **`turbo-lora-backport`** | `4c28e247ecd8ffa9b4507a90c1cf854bb734cc8f` | `4c28e24` | 0 | 2026-08-08 · `[diffusion] fix: fix 4/8-step distilled minimax-h3 turbo lora merge (#33875)`；由上游 `914644e` cherry-pick 而来(`merge-base --is-ancestor` 判否 → 确认是 cherry-pick 不是 merge)，基线 `main` = `407a65d3cb5eb38873b34de75d3bcd76be5a0bee` |
| `~/data/dropbox/CV/h3/ComfyUI/custom_nodes/ComfyUI-MultiGPU` | `pollockjj/ComfyUI-MultiGPU` | `main` | `b51c99a525e9607e43545ee2a8b7694c74a4775a` | `b51c99a` | 0 | 2026-05-08 · `Merge pull request #199 from pollockjj/codex/aimdo-device-fallback`；**无 `requirements.txt`(零 pip 依赖)**，DisTorch2 实现在 `distorch_2.py` |

> **档案缺口已补 / 冲突点**：
> 1. 历史档案写「ComfyUI-MultiGPU / DisTorch2 节点版本 pin 档案里没记」——现记为 **`b51c99a525e9607e43545ee2a8b7694c74a4775a`**。
> 2. 历史档案写 6000a 的 `custom_nodes` 有多个节点——**实测 `custom_nodes/` 下只有 `ComfyUI-MultiGPU` 一个 git 仓库**(外加一个 `__pycache__`)。
>    **TeaCache 节点与 Turbo-LoRA 作者节点只在 5090 上，本机没有**。要在 6000a 复现 ComfyUI Turbo 路线必须先 clone 它们。
> 3. 历史档案写 6000a 的 `src/vllm-omni-pr5910` 是 `@ b18eeff2 + stride patch`。**实测它就在分支
>    `pr5910-stride-fix` 的 `070096bd` 上**(补丁已固化成 commit，父提交才是 `b18eeff2`)，
>    与 5090 的固化分支同一个 commit。表述"b18eeff2 + patch"在字节层面等价，但**恢复时应该
>    `git switch pr5910-stride-fix`，不要去 checkout 裸的 b18eeff2 再手打补丁**。

### 4.2 补丁表(**按代码内容判定，不看 `git diff`**)

补丁一旦 commit，worktree 就是干净的，`git diff --quiet` 会反过来报"没打"。以下判定命令都是内容级的。

| # | 补丁 | 改哪里 | 改什么 / 为什么 | 不打的后果 | 判定命令 | **本机实测结果** |
|---|---|---|---|---|---|---|
| 1 | **PR5910 resident stride fix** | `src/vllm-omni-pr5910/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py`，`class PinnedResidentLayerGroup.load()` | 把 `gpu_buffer[off:off+numel].view(meta["shape"])` 换成 `torch.as_strided(…, size=meta["shape"], stride=meta["stride"])`。`_shard_and_pin` 按物理(列主序)顺序打包非连续权重并存了 stride，streamed 路径 `prefetch_layer` 用 `as_strided` 还原得对，常驻组却用 `.view()` 按行主序重解释 | **不报错，静默产纯噪声**(宽松 kernel)或**崩溃**(严格 kernel) | `sed -n '/class PinnedResidentLayerGroup/,/def offload/p' <文件> \| grep -q as_strided` —— **必须限定在类内**，因为文件别处(streamed 路径)本来就有 `as_strided`，裸 `grep as_strided <文件>` 会恒真 | **已打**：类内命中 `torch.as_strided(`，全文件 `as_strided` 出现 **4** 次 |
| 1' | 同上，对照 | `src/vllm-omni/…/distributed_layerwise_backend.py`(main `a874b8e`) | — | — | 同上 | **未打**：全文件 `as_strided` 出现 **0** 次，类内是 `gpu_buffer[…].view(meta["shape"])`。**这是预期的** —— 6000a 生产走 main 且**不开 DLO**，走不到这条常驻路径。但若有人在 main 上打开 DLO，会踩到同一个 bug |
| 2 | **SGLang 2D fused `lora_B` 的 TP 切片** | `src/sglang/python/sglang/multimodal_gen/runtime/layers/lora/linear.py`，`MergedColumnParallelLinearWithLoRA.slice_lora_b_weights`(commit `4c28e24`，改动 25 行) | 原实现只处理 3D stacked `lora_B`(`B[:, start:end, :]`)。H3 的原生 fused checkpoint 每个逻辑层是**一个拼接的 2D `lora_B`**，需要按 `output_sizes` / `output_partition_sizes` 逐段独立切分再 `cat`。新代码用 `if B.dim() == 3` 分派 | `sglang-lora`(dynamic LoRA)路线 **IndexError**，整条蒸馏 AB 路线不可用 | `git -C src/sglang rev-parse --abbrev-ref HEAD`(应为 `turbo-lora-backport`)+ `grep -n "B.dim() == 3" python/sglang/multimodal_gen/runtime/layers/lora/linear.py`；另有单测 `scripts/test_sglang_lora_slice.py`(TP1/2/4) | **已打**：分支 `turbo-lora-backport` @ `4c28e24`，`linear.py` 含 `if B.dim() == 3:` 与 `for full_size, part_size in zip(self.base_layer.output_sizes, self.base_layer.output_partition_sizes)` |
| 3 | **引擎级 30 秒硬超时** `_ASYNC_OUTPUT_TIMEOUT` | `vllm_omni/diffusion/diffusion_engine.py:58` | 写死 `_ASYNC_OUTPUT_TIMEOUT = 30.0`，在 `step_streaming` 里作为"等引擎吐下一个输出"的上限；超时抛 `TimeoutError`，被包成 HTTP 500，body 是**没有内容的** `{"error":{"message":"Video generation failed:"}}` | 慢档位(尤其 BF16 基座高 NFE)会**假失败**，且错误信息为空，极易误判成显存/配置问题 | `grep -n "_ASYNC_OUTPUT_TIMEOUT" <checkout>/vllm_omni/diffusion/diffusion_engine.py` | **两份 checkout 都仍是 `30.0`(未打)**：`src/vllm-omni` 与 `src/vllm-omni-pr5910` 的第 58 行均为 `_ASYNC_OUTPUT_TIMEOUT = 30.0  # seconds` |

> ⚠️ **冲突点(重要)**：历史档案 2.2 把 `_ASYNC_OUTPUT_TIMEOUT` 列为"6000a 必须打的补丁之三"。
> **实测本机根本没打**。也就是说：档案里的"必打"是**结论/建议**，不是**现状**。
> 若后续在 6000a 上跑 BF16 基座高 NFE 而遇到 500 + 空 message，**先去改这一行(editable 安装，改完重启引擎即可)**，
> 不要再从显存方向排查。注意它与 `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800` **是两回事**，后者管不到它。

### 4.3 其他与代码相关的实测事实

- SGLang 的 **768 短边硬锁**确实在源码里：
  `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py:88`
  抛 `"{path}.short_edge must be 768 for minimax_h3, got {short_edge}"`，
  `resolved_plan.py:103,106` 也各有一处。→ 本 commit 下 SGLang 只能做 768P 高清档，
  `h3_generate.py` 里 sglang profile 不传 canvas 就是因为传了也没用。
- `4c28e24` 相对 `407a65d` 的完整 diff stat：`docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx`(+28/-4)、
  `compatibility_matrix.mdx`、`quantization.mdx`、**`multimodal_gen/runtime/layers/lora/linear.py`(+25/-4)**、
  `test/unit/test_lora_format_adapter.py`(+12)。**只有 `linear.py` 是功能改动。**
  (档案把它写成 `MergedColumnParallelLinearWithLoRA.slice_lora_b_weights` 是对的，但**文件不是
  `srt/lora/layers.py`**——那是 LLM 侧的同名类，diffusion 侧在 `multimodal_gen/runtime/layers/lora/linear.py`。
  改错文件不会报错，只会白改。)

---

## 5. 模型权重

全部权重在第二块 NVMe `/home/isaac/Data/h3_weights/`，ComfyUI 侧用 symlink 引回 ——
**除了 INT8 两件 + 两个 VAE(video fp16 / audio fp32)共 4 个文件是实体落在根盘**(见下表 ⚠️ 标记)。
实测:`ComfyUI/models/{diffusion_models,text_encoders}` 下 BF16 是 symlink、INT8 是实体;
`ComfyUI/models/vae/` 下两个文件**都是实体,没有 symlink**(因为 `download_h3.sh` 直接下到 ComfyUI 目录,不建 symlink)。

| 路径 | 体积 | 顶层内容 | 来源(HF repo @ revision) | 用途 | 校验契约 |
|---|---|---|---|---|---|
| `/home/isaac/Data/h3_weights/MiniMax-H3` | **145G**(其中 FL2VA 子树 135G，**另有 11G 是残留 `.incomplete`**，见 §8) | 完整 HF 布局：`FL2VA/` `Ref2VA/` `transformer/` `transformer_ref/` `vae/` `text_encoder/` `tokenizer/` `processor/` `scheduler/` `audio_scheduler/` `docs/` `model_index.json` `modular_model_index.json` `README.md` | `MiniMaxAI/MiniMax-H3` @ **`101ecd0aa25532b6443c625340f095a617a2526a`**(实测：`.cache/huggingface/download/**/*.metadata` 共 **138** 个文件，第一行 commit hash **全部**是这一个) | vLLM `vllm bf16/fp8`、SGLang `sglang`、Turbo merge 的 base | `hf download` 的 etag/大小校验；`FL2VA/transformer` **13 shard**、`FL2VA/text_encoder` **14 shard** |
| ├ `MiniMax-H3/FL2VA/transformer` | 62G | 13 × `model-000NN-of-00013.safetensors` + `config.json` + `model.safetensors.index.json` | 同上 | DiT 主干；`config.json`: `num_layers=50`、`hidden_size=5376`、`num_attention_heads=56`、`attention_head_dim=128`、`ffn_hidden_size=14336`、`adaln_out_features=96768` | — |
| ├ `MiniMax-H3/FL2VA/text_encoder` | 63G | 14 shard + index | 同上 | Qwen3-VL-32B TE | — |
| ├ `MiniMax-H3/FL2VA/video_vae` | 9.8G | 权重 + **一批 `.py` 源码**(`minimax_h3_video_vae.py` 等)+ `source/` | 同上 | 需 `--trust-remote-code` | — |
| └ `MiniMax-H3/FL2VA/audio_vae` | 578M | `model.safetensors` + `.py` + `config.yaml` | 同上 | 音频 VAE | — |
| `…/h3_weights/diffusion_models/minimax_h3_fl2va_bf16.safetensors` | **66280487368 B**(66.3G) | 单文件 | `Comfy-Org/MiniMax-H3` `resolve/main`(**未记 revision，只按字节数校验**) | ComfyUI `comfy-bf16-original-4c` | `download_bf16.sh` 按**精确字节数**判完成；symlink → `ComfyUI/models/diffusion_models/` |
| `…/h3_weights/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | **51506295256 B**(51.5G) | 单文件 | 同上 | 同上 | 同上 |
| `ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` ⚠️**实体在根盘** | **20970379616 B**(20G) | 单文件 | `Comfy-Org/MiniMax-H3`(未记 revision) | ComfyUI `comfy-int8-original-1c` | `download_h3.sh` 字节数校验 + curl `-C -` 续传 |
| `ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` ⚠️**实体在根盘** | **27141342152 B**(26G) | 单文件 | 同上 | 同上(6000a profile 的默认 TE) | 同上 |
| `ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors` ⚠️**实体在根盘** | **5207808496 B** | 单文件 | 同上 | ComfyUI 全部 profile | 同上 |
| `ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors` ⚠️**实体在根盘** | **605254808 B** | 单文件 | 同上 | ComfyUI 全部 profile | **fp32 是硬性要求**(BF16 会 −20dB) |
| `…/h3_weights/loras/minimax_h3_turbo_v4_step600_ema.safetensors` | **779849816 B** | 单文件 | `larryvrh/MiniMax-H3-Turbo-Lora` @ **`afc0346516372a17162c14df3c5264de1d9aa1c0`**；**sha256 `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`** | Turbo merge 的输入；`h3_switch.sh sglang-lora` 的 `--lora-path` | `.cache/huggingface/download/*.metadata` 记 revision+sha256；G0 体检 `scripts/check_turbo_lora.py <lora路径>`(**位置参数,无 `--lora`**;259 对键 / 518 tensors / 全 BF16 / shape 逐表核对) |
| `…/h3_weights/loras/minimax_h3_turbo_4step_ema_ckpt850.safetensors` | **779849816 B** | 单文件 | 同 repo 同 revision；**sha256 `5a6eeba171cf183020a4ad48774bb2968f29f8168afd6ec17a04987f3528b4ea`** | 大运动备选(AB 用) | 同上 |
| `…/h3_weights/MiniMax-H3-Turbo-v4s600ema` | **62G** | `FL2VA/transformer/` 是**实体**(13 shard，权限 `-rw-------`)；其余顶层项全是 symlink 回 base；根下有 `merge_manifest.json`、**`.complete`**、`delta_norms.csv` | 本地 merge 产出(见下) | vLLM `vllm-turbo bf16/fp8`、SGLang `sglang-turbo` | `h3_switch.sh` 的 `require_complete()`：**根目录缺 `.complete` 就拒绝启动**。`.complete` 内容 = `2026-08-08T15:58:07+1000` |
| `…/h3_weights/turbo_v4s600ema_alias/MiniMax-H3` | symlink | → merged 根 | — | **SGLang 专用别名**：native H3 识别按路径 **basename 短名 == `MiniMax-H3`**，`h3_switch.sh` 自动建 | — |

**根盘压力**：`ComfyUI/models` 实测 **51G**(int8 DiT 20G + int8 TE 26G + VAE 5.5G)落在 94% 满的根盘上，
且位于 Dropbox 同步目录 `~/data/dropbox/` 之内。BF16 单文件与官方 HF 权重则在第二块 NVMe。

### 5.1 merged Turbo 权重的 `merge_manifest.json`(实测全文关键字段)

```json
{
  "base_model": "MiniMaxAI/MiniMax-H3",
  "base_path": "/home/isaac/Data/h3_weights/MiniMax-H3/FL2VA",
  "base_transformer_index_sha256": "fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb",
  "lora_repo": "larryvrh/MiniMax-H3-Turbo-Lora",
  "lora_revision": "afc0346516372a17162c14df3c5264de1d9aa1c0",
  "lora_file": "minimax_h3_turbo_v4_step600_ema.safetensors",
  "lora_sha256": "5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3",
  "strength": 1.0,
  "alpha_convention": "absent -> alpha=rank (scale 1.0)",
  "merge_dtype": "float32",
  "output_dtype": "bfloat16",
  "qkv_disk_layout": "group-interleaved (56 groups x [q|k|v] x 128)",
  "qkv_runtime_layout": "Q_all_K_all_V_all (LoRA lora_B order)",
  "merge_script_sha256": "b034c41b1c20233333890d5c168583de48730fa9f7df10c2c4041c615b5dcb1b",
  "modified_tensor_count": 259,
  "total_tensor_count": 535,
  "created": "2026-08-08T15:58:07+1000",
  "verification": "L1 bit-level recompute; L2 Comfy single-file layout oracle; L3 single-layer forward oracle (cos>0.9999, relL2<5e-3)"
}
```

`logs/merge_turbo_lora.log` 里的**实测校验数值**(比 manifest 的阈值更严)：

- L1：`259 modified + 276 unmodified = 535 tensors bit-checked`，13 个 shard 逐个通过
- L2：`Comfy single-file == reorder(HF disk) bit-exact on 6 qkv + 5 plain layers`
- L3：`worst min_row_cos=0.999999, worst relL2=9.65e-04`(最差在 `blocks.49.attn.qkv_proj.weight`)
- `delta_norms.csv` top-10：最大 `blocks.49.mlp.fc2.weight = 0.0036`，其余均 ≤0.0023，集中在 blocks 43–49
- 耗时：13 个 shard 各 ~9s 合并 + 各 ~5s L1 复算，整体约 3 分钟(15:55:01 → 15:58:07)

> ⚠️ 这三层校验里 **L2 需要 Comfy 单文件做独立布局 oracle**。5090/runpods 是靠"溯源哈希与 6000a 完全一致"
> 替代 L2 的(`--comfy-single /nonexistent` 跳过)。**6000a 是唯一真跑过 L2 的机器** —— 因为只有它同时有
> HF 分片权重和 Comfy 单文件权重。

### 5.2 HF cache 桥接(让 SGLang 免于重下 145G)

```
~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/
├── refs/main                                   → 内容 "b3c7290e66afdf293bef3b9077b7a266ef421f34"
└── snapshots/b3c7290e66afdf293bef3b9077b7a266ef421f34/
    ├── model_index.json → /home/isaac/Data/h3_weights/MiniMax-H3/model_index.json   (symlink)
    ├── FL2VA/transformer/model-00001-of-00013.safetensors → …(逐文件 symlink)
    └── …
```
该目录**没有 `blobs/`**，是纯手工桥接(不是 hf 真缓存)。`h3_switch.sh sglang` 传的 model path 就是
仓库名 `MiniMaxAI/MiniMax-H3`，靠这个桥接命中本地文件。

> ⚠️ **冲突点(重要，provenance 风险)**：桥接目录名与 `refs/main` 写的是
> **`b3c7290e66afdf293bef3b9077b7a266ef421f34`**(与历史档案一致)，
> 但**实际下载下来的 138 个文件的 `.metadata` 记录的 commit 全部是
> `101ecd0aa25532b6443c625340f095a617a2526a`**。
> 两者不一致 —— 桥接对外**宣称**的 revision 与磁盘字节的**真实**revision 不是同一个。
> 功能上无碍(SGLang 只按 `refs/main` → snapshot 目录解析)，但**溯源时以 `101ecd0…` 为准**；
> 下载时间(Aug 5 23:44 起)早于桥接创建(Aug 6 13:41)，**推测**是桥接创建时抄了当时 HF 上的
> 最新 main sha 而非下载时的 sha(依据：两个时间戳的先后 + 138/138 metadata 高度一致)。
> 要复现"位级相同的 base"，应该用 `--revision 101ecd0aa25532b6443c625340f095a617a2526a`。

LoRA 侧同理有一个只含 `refs/main` 的目录 `models--larryvrh--MiniMax-H3-Turbo-Lora/refs/main = afc0346516372a17162c14df3c5264de1d9aa1c0` —— 这个与实际下载 metadata **一致**。

---

## 6. 运行期环境变量

全部来自本机 `scripts/h3_switch.sh` 与 `scripts/launch_comfy_6000a_4gpu.sh`(远端 worktree 干净，与 Mac 仓库同 HEAD)。

### 6.1 vLLM-Omni(`h3_switch.sh` → `start_vllm`)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
```

| 变量 | 作用 |
|---|---|
| `CUDA_VISIBLE_DEVICES=0,1,2,3` | TP4 需要四卡全可见 |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | 多进程 worker 用 spawn 而非 fork，避免 CUDA context 在 fork 后不可用 |
| `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800` | 同步视频接口的请求上限 30 分钟。**管不到 `_ASYNC_OUTPUT_TIMEOUT=30.0`**(见 §4.2 补丁 3) |

serve flags(实测运行中的进程命令行与脚本一致)：
```
vllm serve <model_root>/FL2VA --omni --host 127.0.0.1 --port 8091 --trust-remote-code
  --num-gpus 4 --tensor-parallel-size 4 --usp 1 --ring 1 --text-encoder-tp-size 4
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling
  --diffusion-attention-backend FLASH_ATTN  [--quantization fp8]
```
**本机不开 DLO、不设 `VLLM_DISABLED_KERNELS`、不设 `PYTORCH_CUDA_ALLOC_CONF`** ——
这三样分别是 5090(sm_120 禁 Cutlass)和 runpods(expandable_segments)的机器特化，
6000a 是 sm_89，Cutlass FP8 可用，不需要。attention backend 也不同：**6000a 用 `FLASH_ATTN`，
5090/runpods 用 `CUDNN_ATTN`**。

### 6.2 SGLang(`h3_switch.sh` → `start_sglang`)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_HOME=$HOME/miniconda3/envs/h3_sglang_NV_py312/lib/python3.12/site-packages/nvidia/cu13
PATH=$SGL_BIN:$CUDA_HOME/bin:$PATH          # SGL_BIN=…/envs/h3_sglang_NV_py312/bin(ninja 在这)
LIBRARY_PATH=$CUDA_HOME/lib
LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
RUNAI_STREAMER_MEMORY_LIMIT=8589934592      # 8 GiB
NINJA_STATUS="[ninja] "
```

| 变量 | 作用 |
|---|---|
| `CUDA_HOME` / `PATH` / `LIBRARY_PATH` / `LD_LIBRARY_PATH` | 系统无 CUDA toolkit，SGLang 的 JIT 需要 `nvcc`、`-lcudart`、运行时 so。全部指向 pip 装的 `nvidia/cu13`。**`libcudart.so` 无版本 symlink 必须存在**，否则 `-lcudart` 链接失败 |
| `RUNAI_STREAMER_MEMORY_LIMIT=8589934592` | runai model streamer 的 CPU 暂存上限。4 个 rank 并发加载不设限会把 376G RAM 打爆 |
| `NINJA_STATUS` | 只是让 JIT 编译日志好认，无功能影响 |

serve flags：
```
sglang serve --model-path <path> --model-variant fl2va --num-gpus 4 --tp-size 4
  --ulysses-degree 1 --performance-mode speed --enable-torch-compile false
  --quantization fp8 --host 127.0.0.1 --port 30010
# sglang-lora 追加: --lora-path <turbo lora> --lora-nickname turbo --lora-scale 1.0 --lora-merge-mode dynamic
```
`--quantization fp8` 是**固化的**：768P + TP4 的 BF16 在 48G 卡上 OOM。
`--lora-merge-mode dynamic` 必须**显式**给，`auto` 会把 LoRA merge 进 FP8 权重(禁忌)。

### 6.3 ComfyUI(`launch_comfy_6000a_4gpu.sh`)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
LD_LIBRARY_PATH=$HOME/miniconda3/envs/h3_comfy_NV_py312/lib/python3.12/site-packages/nvidia/cu13/lib
# 启动: setsid nohup $ENVPY main.py --listen 127.0.0.1 --port 8288 \
#         --output-directory $BASE/outputs --input-directory $BASE/inputs
```
四卡全可见是给 DisTorch2 跨卡分置用的(`comfy-bf16-original-4c` 的
`--dit-compute cuda:2 --dit-vvram 35 --dit-donor cuda:3 --te-compute cuda:0 --te-vvram 30 --te-donor cuda:1 --vae-device cuda:1`
由客户端 `run_fl2va.py` 转成节点参数)。`LD_LIBRARY_PATH` 指向 env 内的 cu13 lib(`libcublas.so.13` 等)。
脚本开头有幂等短路：`/system_stats` 能通就直接 `echo "already running"; exit 0`。

### 6.4 采集期未见的环境变量

非登录 shell 下 `env | grep -E "^HF_|^HUGGING"` 为空 —— **HF token 不走环境变量**，
放在 `~/.cache/huggingface/token`(文件存在，**未读取其内容**)。
下载脚本 `dl_fl2va_auth.sh` 自己 `export HF_HUB_ENABLE_HF_TRANSFER=1`(已废弃，见 §8)。

---

## 7. 从零复现

目标：在一台同规格(4×48G sm_89、≥376G RAM、driver ≥580.65)的干净 Ubuntu/Pop!_OS 22.04 上，
把 6000a 的全部 8 个 profile 重建出来。**每步都写了"为什么"**。

### 步骤 0 — 系统前提

```bash
# 校验驱动:必须 >= 580.65 才能跑 CUDA 13.0 wheel
nvidia-smi --query-gpu=driver_version,name,memory.total,compute_cap --format=csv
# 期望: 580.173.02, NVIDIA RTX 6000 Ada Generation, 49140 MiB, 8.9  (×4)

# 校验 P2P:必须全 OK,否则 DisTorch2 跨卡与 TP4 的假设不成立
nvidia-smi topo -p2p r      # 期望四卡两两 OK,不能有 CNS

sudo apt-get install -y build-essential cmake git ffmpeg
# gcc 11.4 即可,不需要系统 CUDA toolkit;本机实测 ffmpeg = 4.4.2-0ubuntu0.22.04.1(系统包,不是 conda)
which ffmpeg ffprobe flock ss     # flock 来自 util-linux、ss 来自 iproute2,Ubuntu/Pop 基础安装自带;缺了再补装
```

> ⚠️ **`ffmpeg`/`ffprobe` 两侧都要**:远端 `scripts/probe.sh:13`、`bench_matrix.sh:39`、
> `extract_frames.sh:15-16`、`tp2_r2_probe.sh:15-16` 直接调 `ffmpeg`;Mac 客户端
> `h3_generate.py:563` 与 `h3_eval.py:199` 有 `for tool in ("ffmpeg","ffprobe","ssh","scp")` 的硬前置检查
> (Mac 侧还要 ffmpeg 做画布归一化,`h3_generate.py:336`)。缺了整条 benchmark/probe/抽帧链路会挂。
> `h3_switch.sh` 另外用到 `flock`(:142-143)与 `ss`(:44)。

> ⚠️ **不要装系统 CUDA toolkit**。本机全靠 pip 的 `nvidia-cuda-nvcc`/`nvidia-cuda-runtime` 提供 nvcc 与
> libcudart，装了系统 toolkit 反而可能让 SGLang JIT 链到版本不匹配的库。

**克隆项目仓库并建运行时目录**(后面步骤 5/6/7/8 引用的 `$BASE/scripts/*` 全靠它;
`logs/ outputs/ run/ inputs/` 都在 `.gitignore` 里,clone **带不出来**,必须手动建):

```bash
BASE=$HOME/data/dropbox/CV/h3          # 这就是 $BASE = git 工作区(本机 HEAD 39c4b5089f78f25ae62b5df19fa2e5f9492768c8)
git clone https://github.com/stargazerscripter123/H3_inference.git "$BASE"
mkdir -p "$BASE"/{inputs,outputs,workflows,logs,run}
```
**为什么必须先建目录**:`launch_comfy_6000a_4gpu.sh` 带 `set -euo pipefail` 且重定向到
`$BASE/logs/comfyui.log`(全脚本无 `mkdir`);`h3_switch.sh` 重定向到 `$BASE/logs/{vllm,sglang}_server.log`
(只 `mkdir -p "$BASE/run"`,不建 `logs/`)。目录不存在则启动**直接失败**。
仓库自带的 `scripts/setup_h3.sh:12` 就是这一步(`mkdir -p "$BASE"/{inputs,outputs,workflows,scripts,logs}`)。

> 💡 也可以直接 `bash scripts/setup_h3.sh` —— 它一口气做了本节步骤 0 的建目录 + 步骤 1 的 comfy env +
> 步骤 2 的 torch/ComfyUI pin/requirements。**它没覆盖的差异**:不建 `run/`、不装 MultiGPU 节点、
> 不建 vllm/sglang 两个 env(步骤 1 的另两个)、且第 39 行仍在装已废弃的 `hf_transfer`(见步骤 2 注)。

> ⚠️ **磁盘规划先做**：稳态 = 官方权重 145G + merged 62G + BF16 单文件 118G + INT8 组 51G ≈ **376G**；
> **下载期另需 ≥15G 暂存**(`hf download --local-dir` 先在 `.cache/huggingface/download/` 落 `.incomplete` 再改名，
> 本机至今还留着 11G 没清，见 §8 第 1 条)。**建议大盘至少留 450G 可用**。
> 本机把大件放第二块 NVMe(`/home/isaac/Data/h3_weights`)，只有 INT8 组 + 两个 VAE 落在根盘 ——
> 两块盘现在都已 93–94% 满(`/` 余 244G、`/home/isaac/Data` 余 284G)。
> 复现时**建议把 `ComfyUI/models` 也整体 symlink 到大盘**，避免重蹈覆辙。

### 步骤 1 — 三个 conda 环境

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
for E in h3_comfy_NV_py312 h3_vllm_NV_py312 h3_sglang_NV_py312; do
  conda create -y -n $E python=3.12
done
```
**为什么分三个 env**：三条产线对 `transformers`/`diffusers` 的版本要求不同(实测 comfy 5.14.1 /
vllm 5.14.1+diffusers 0.38.0 / sglang 5.12.1+diffusers 0.37.0)，装一起必然互相降级。
**为什么 3.12**：ComfyUI pin 与 vllm 0.26.0 / sglang 都验证在 3.12(实测三个 env 都是 3.12.13)。

### 步骤 2 — ComfyUI env

```bash
BASE=$HOME/data/dropbox/CV/h3
PIN=57500fc5bc92566a63f2046824f522cd55c335ca
ENVPY=$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python

$ENVPY -m pip install torch==2.11.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130     # -> 2.11.0+cu130 / 0.26.0+cu130 / 2.11.0+cu130

mkdir -p "$BASE/ComfyUI" && cd "$BASE/ComfyUI"
git init -q && git remote add origin https://github.com/comfyanonymous/ComfyUI.git
git fetch --depth 1 origin "$PIN" && git checkout -q FETCH_HEAD    # -> detached, __version__ 0.29.0

$ENVPY -m pip install -r requirements.txt          # 内含 comfy-kitchen==0.2.26 / comfy-aimdo==0.4.11
$ENVPY -m pip install "huggingface_hub[cli]"       # 装出来的命令叫 hf(不是 huggingface-cli),见步骤 6

# 4 卡 oracle 需要 MultiGPU / DisTorch2
cd "$BASE/ComfyUI/custom_nodes"
git clone https://github.com/pollockjj/ComfyUI-MultiGPU.git
git -C ComfyUI-MultiGPU checkout b51c99a525e9607e43545ee2a8b7694c74a4775a   # 无 requirements.txt,零 pip 依赖
```
**为什么钉 `57500fc`**：这是 ComfyUI 支持 MiniMax-H3 的那个 merge commit(`feat: Support MiniMax-H3 (CORE-375) (#15224)`，v0.29.0)。跟 master 会漂。
**为什么 cu130**：驱动 580.173.02 支持 CUDA 13.0(见 §2)。
**为什么必须装 comfy-kitchen / comfy-aimdo**：INT8/量化算子在这两个包里，缺了 INT8 profile 直接跑不起来；它们本来就写在 pin 版 `requirements.txt` 里，不要手动升级。
**为什么不装 `hf_transfer`**：本机历史上装过(`hf_transfer 0.1.9` 还在 comfy env 的 freeze 里)，
`scripts/setup_h3.sh:39` 也还在装，但 huggingface_hub 1.26 已经**不再使用**它(见 §8 第 4 条的 FutureWarning 原文)。
复现不需要，统一用 `HF_XET_HIGH_PERFORMANCE=1`(`hf-xet 1.6.0` 由 `huggingface_hub[cli]` 自带)。

> ⚠️ 本机 `custom_nodes` **只有 MultiGPU**。ComfyUI 侧的 TeaCache / Turbo-LoRA 作者节点是 5090 的资产，
> 6000a 从来没装。如果要在 6000a 上复现 ComfyUI Turbo 路线，需另行 clone
> `Icyoung/ComfyUI-MiniMaxH3-TeaCache@4cbb50d` / `Larryvrh/ComfyUI-MiniMax-H3-Turbo@55fee864`(版本来自历史档案，本机**未实测**)。

> ⚠️ **绝不能用 ComfyUI 核心 `LoraLoaderModelOnly` 加载 Turbo LoRA**：`comfy/lora.py` 的
> `model_lora_keys_unet` 没有 MiniMaxH3 分支，裸键 LoRA 会 **0/518 命中且不报错**，出片等于"低步数的 base"。

### 步骤 3 — vLLM env

```bash
VLM_PY=$HOME/miniconda3/envs/h3_vllm_NV_py312/bin/python
$VLM_PY -m pip install -q uv
$VLM_PY -m uv pip install --python "$VLM_PY" "vllm==0.26.0" --torch-backend=auto

SRC=$HOME/data/dropbox/CV/h3/src && mkdir -p "$SRC"
git clone https://github.com/vllm-project/vllm-omni.git "$SRC/vllm-omni"
git -C "$SRC/vllm-omni" checkout a874b8e09bb5251bb16e3de8bc22edf3880eec09
$VLM_PY -m uv pip install --python "$VLM_PY" -e "$SRC/vllm-omni"
```
**为什么 `vllm==0.26.0`**：与 vllm-omni 0.26.x 系列配对，版本错开会在 import 期报兼容问题。
**为什么 vllm-omni 走源码 editable 而不是 wheel**：**[推测/上游依据,本机未实测]**
`vllm-omni==0.26.0` 的 wheel 里 H3 据信**缺首尾帧 `frame_indices` 支持**(8-05 才 merge)——
本项目是 FL2VA，没有它整条产线无意义。
**注意本机日志并不能证实这一点**:wheel 时期的 `logs/install_backends2.log` 只记了
`vllm 0.26.0 | vllm_omni 0.26.0` 与 `minimax_h3 present: True`,**从没对 wheel 做过 frame_indices 检查**;
下面那条 `YES` 是 editable 装**之后**打的。要坐实这条结论，需在临时 env 里装 wheel 后自行 grep 复核。
装完源码版的自检(本机 `logs/install_vllm_src.log` 有留档)：
```
vllm_omni path: …/src/vllm-omni/vllm_omni
minimax_h3 frame_indices support: YES
```
> 对照:同段对 sglang 的论断**是有实测支撑的** —— `install_backends2.log` 里确有
> `sglang 0.5.16` + `ImportError: cannot import name 'minimax_h3'`(见步骤 4)。
**为什么 editable**：`_ASYNC_OUTPUT_TIMEOUT` 之类的本地改动要能直接生效(见步骤 6)。

> ⚠️ 先装 `vllm` 再装 vllm-omni editable —— vllm-omni 的构建脚本 import 自身依赖，顺序反了会失败。
> 若源码是 rsync 过来且丢了 `.git`，setuptools_scm 会把版本算成 `dev` 导致构建失败，
> 此时用 `SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0 pip install -e . --no-build-isolation`。

### 步骤 4 — SGLang env

```bash
SGL_PY=$HOME/miniconda3/envs/h3_sglang_NV_py312/bin/python
$SGL_PY -m pip install -q uv
git clone https://github.com/sgl-project/sglang.git "$SRC/sglang"
git -C "$SRC/sglang" checkout 407a65d3cb5eb38873b34de75d3bcd76be5a0bee
cd "$SRC/sglang"
SGLANG_BUILD_RUST_EXTS=none $SGL_PY -m uv pip install --python "$SGL_PY" \
    -e "python[diffusion]" --prerelease=allow

# JIT 依赖(系统无 CUDA toolkit)—— 必须钉版本,见下方 ⚠️
$SGL_PY -m pip install --only-binary :all: nvidia-cuda-nvcc==13.3.73 nvidia-cuda-runtime==13.0.96
CUROOT=$HOME/miniconda3/envs/h3_sglang_NV_py312/lib/python3.12/site-packages/nvidia/cu13
ls "$CUROOT/bin/nvcc"                                # 先确认落点确实是 nvidia/cu13/
CU="$CUROOT/lib"
ln -sfn "$CU/libcudart.so.13" "$CU/libcudart.so"     # 无版本 symlink,-lcudart 要它
```
**为什么源码装**：`sglang==0.5.16` 的 wheel **没有 `multimodal_gen` 的 H3 pipeline config**
(本机 `logs/install_backends2.log` 留着那次失败：`ImportError: cannot import name 'minimax_h3'`)。
**为什么 `SGLANG_BUILD_RUST_EXTS=none`**：Rust 扩展与本产线无关，编译它只会增加失败面。
**为什么 `--prerelease=allow`**：`[diffusion]` extra 依赖里有 prerelease 包(如 `flash-attn-4 4.0.0b19`)。

> ⚠️ **nvcc 系的包要用 `--only-binary :all:`**。把它们和别的包合并成一行装，一旦落到 sdist 构建
> 就会连带把 uv/ninja 一起搞崩，后面所有步骤静默不装(5090 上踩过，修复脚本是 `scripts/fix_5090_installs.sh`)。

> ⚠️ **必须钉版本,因为 §6.2 的 `CUDA_HOME` 把路径硬写死到 `site-packages/nvidia/cu13`**。
> 本机实测 `nvidia-cuda-nvcc==13.3.73` / `nvidia-cuda-runtime==13.0.96`(freeze 里全套 nvidia 包都是
> 无后缀的 CUDA 13 新命名:`nvidia-cublas`/`nvidia-nvjitlink`/`nvidia-nvvm` 等)。**不钉版本**的话,
> 将来 pip 解析到 cu14 系列会把文件落到 `nvidia/cu14/`,于是 `CUDA_HOME`、`libcudart.so` symlink、
> `LIBRARY_PATH`、`LD_LIBRARY_PATH` 全部指空 —— SGLang JIT 会以**链接失败**或**静默走错库**收场。
> 装完先 `ls $CUROOT/bin/nvcc` 确认落点,若不是 `nvidia/cu13` 就要同步改 §6.2 的四个路径变量。
> **与 5090 脚本的包名关系**:`scripts/fix_5090_installs.sh:26` 用的是带后缀的
> `nvidia-cuda-nvcc-cu13 nvidia-cuda-runtime-cu13`(旧式 per-CUDA-版本包名),
> 与本机 freeze 里的无后缀名是**同一套 CUDA 13 二进制的两种发布名**;
> **6000a 复现请照本节的无后缀名 + 钉版本来装**,以便和 lock 文件逐行对得上。

### 步骤 5 — 打补丁

```bash
# (a) SGLang 2D fused lora_B 的 TP 切片 backport —— dynamic LoRA 路线必需
cd "$SRC/sglang"
git fetch origin
git checkout -b turbo-lora-backport 407a65d3cb5eb38873b34de75d3bcd76be5a0bee
git cherry-pick 914644e81c9b            # 上游 PR #33875
git rev-parse HEAD                      # 本机留档 4c28e247ecd8ffa9b4507a90c1cf854bb734cc8f —— 你的机器上必然不同,见下
# 判定(内容级 + 差异级,不看 git diff --quiet):
git rev-parse --abbrev-ref HEAD         # 期望 turbo-lora-backport
grep -n "B.dim() == 3" python/sglang/multimodal_gen/runtime/layers/lora/linear.py
git diff 407a65d3cb5eb38873b34de75d3bcd76be5a0bee..HEAD --stat
#   期望 5 files changed, +61 −10;其中 multimodal_gen/runtime/layers/lora/linear.py +25 −4(唯一功能改动)
$SGL_PY "$BASE/scripts/test_sglang_lora_slice.py"      # TP1/2/4 单测
```
> ⚠️ **`4c28e247…` 是本机留档,不是期望值**。`cherry-pick` 会写入新的 committer/date,
> 你的机器上算出的 commit hash **必然不同**(lock 文件里的 `-e git+…@4c28e247…` 也会跟着对不上,
> 那是本机 freeze 的快照,不是复现契约)。**判定一律用「分支名 + 内容级 grep + `git diff <base>..HEAD --stat`」。**
> ⚠️ editable 安装下**换分支不用重装**，但 `pip show sglang` 的版本号会停留在安装时的
> `g407a65d3c` 不再变 —— **判断装了什么一律看 `git rev-parse`**。

```bash
# (b) PR#5910 影子 checkout(TP 交叉验证/DLO 路线用;6000a 生产不需要)
git clone https://github.com/vllm-project/vllm-omni.git "$SRC/vllm-omni-pr5910"
cd "$SRC/vllm-omni-pr5910"
git fetch origin pull/5910/head:pr5910-head
git checkout -b pr5910-stride-fix b18eeff22a30b426060f44f734d8b16341b02959
git apply "$BASE/scripts/pr5910_resident_stride_fix.patch"
git commit -am "fix(dlo): keep FP8 transposed-view stride when repointing resident layers"
# 本机留档 HEAD = 070096bd6872418cd8c4e1be18ba1450f301180e —— 你的机器上必然不同(新 committer/date),
# 判定按分支名 + 下面的类内 grep,不要拿 hash 对:
git rev-parse --abbrev-ref HEAD          # 期望 pr5910-stride-fix
git diff b18eeff22a30b426060f44f734d8b16341b02959..HEAD --stat
#   期望 1 file changed, +10 −1,只动 diffusion/offloader/distributed_layerwise_backend.py
# 判定(必须限定在类内!):
sed -n '/class PinnedResidentLayerGroup/,/def offload/p' \
  vllm_omni/diffusion/offloader/distributed_layerwise_backend.py | grep -c as_strided   # 期望 >=1
```
> ⚠️ **判定绝不能用裸 `grep as_strided <文件>`** —— 该文件的 streamed 路径(`prefetch_layer`)本来就有
> `as_strided`，裸 grep **恒真**，会让你以为打了补丁其实没打(`scripts/setup_runpods.sh` 就栽在这)。
> 也**绝不能用 `git diff`** —— 补丁已 commit，worktree 干净，`git diff --quiet` 会反过来报"没打"。
> 这份 checkout 用 `PYTHONPATH` 影子加载(见 `scripts/tp2_r2_6000a.sh`)，**不污染生产 env**。

```bash
# (c) 引擎级 30 秒硬超时 —— 本机当前"没打",但强烈建议打
sed -i 's/^_ASYNC_OUTPUT_TIMEOUT = 30.0/_ASYNC_OUTPUT_TIMEOUT = 600.0/' \
    "$SRC/vllm-omni/vllm_omni/diffusion/diffusion_engine.py"
grep -n "_ASYNC_OUTPUT_TIMEOUT = " "$SRC/vllm-omni/vllm_omni/diffusion/diffusion_engine.py"
```
> ⚠️ 不改它的后果：慢档位(尤其 BF16 基座高 NFE)返回 **HTTP 500 + 空 message**，长得像显存问题其实不是。
> 与 `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800` 无关，后者管不到。**editable 安装改完重启引擎即可生效；
> 每次 `git reset`/升级源码后要重打。**

### 步骤 6 — 下权重

```bash
export HF_XET_HIGH_PERFORMANCE=1              # 不要用 HF_HUB_ENABLE_HF_TRANSFER,见 §8
hf auth login                                 # token 落到 ~/.cache/huggingface/token(本机实测该文件存在)
STORE=/home/isaac/Data/h3_weights             # 大盘

# (a) 官方 FL2VA 子树 —— vLLM/SGLang/merge 的 base
find "$STORE/MiniMax-H3/.cache" -name "*.lock" -delete 2>/dev/null   # 清掉上次被 kill 留下的锁
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/**" \
   --revision 101ecd0aa25532b6443c625340f095a617a2526a \
   --local-dir "$STORE/MiniMax-H3"
# (b) 再拉一遍顶层 config(SGLang 要求根目录是完整 HF 布局,不能只有 FL2VA/)
#     本机当时是逐个列 81 个文件名做的(logs/download_toplevel.log: "Fetching 81 files … 00:13",
#     并伴随 "Ignoring --include since filenames have been explicitly set." 的 UserWarning)。
#     等价且可直接粘贴的写法 —— 用 --exclude 把重件全排掉,只留 config/tokenizer 类小文件:
hf download MiniMaxAI/MiniMax-H3 --revision 101ecd0aa25532b6443c625340f095a617a2526a \
   --local-dir "$STORE/MiniMax-H3" \
   --exclude "FL2VA/*" --exclude "*.safetensors" --exclude "assets/*" --exclude "scripts/*" --dry-run
#   先 --dry-run 看清单(按该 revision 的仓库列表计算应为 89 个文件、总量约 60 MB),确认后去掉 --dry-run 再跑
# (c) ComfyUI 单文件(字节数校验 + 断点续传)
bash "$BASE/scripts/download_h3.sh" 6000a      # INT8 DiT + INT8 TE + video VAE fp16 + audio VAE fp32
bash "$BASE/scripts/download_bf16.sh"          # BF16 DiT 66.3G + BF16 TE 51.5G -> 大盘 + symlink
# (d) Turbo LoRA(只拉需要的文件,整仓 22 个文件 = 白下 22G)
hf download larryvrh/MiniMax-H3-Turbo-Lora \
   --revision afc0346516372a17162c14df3c5264de1d9aa1c0 \
   --include "minimax_h3_turbo_v4_step600_ema.safetensors" \
   --local-dir "$STORE/loras"
sha256sum "$STORE/loras/minimax_h3_turbo_v4_step600_ema.safetensors"
# 期望 5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3
$ENVPY "$BASE/scripts/check_turbo_lora.py" "$STORE/loras/minimax_h3_turbo_v4_step600_ema.safetensors"
# ^ lora 路径是【位置参数】,没有 --lora(scripts/check_turbo_lora.py:44 是 p.add_argument("lora");
#   加了 --lora 会直接 "error: unrecognized arguments" 退出)。
#   注意别和下面步骤 7 的 merge_turbo_lora.py 混了 —— 那个才是 --lora/--base/--dst/--strength 全带前缀的。
# 期望 G0 通过: 259 对键 / 518 tensors / 全 BF16 / shape 逐表核对
```
**为什么钉 revision `101ecd0…`**：这是本机磁盘上那份权重的真实 commit(138/138 metadata 实测)。
不钉的话拿到的是当时的 main，与 merged checkpoint 的 `base_transformer_index_sha256` 对不上，
Turbo merge 的溯源链就断了。
**为什么 audio VAE 必须 fp32**：BF16 会让音量掉约 20 dB。

> ⚠️ **(b) 的 `--exclude` 列表不能只排 `FL2VA/` 和 `Ref2VA/`**。该 repo 的**顶层
> `transformer/` `transformer_ref/` `text_encoder/` `vae/` 里各自还有整套 safetensors 分片**
> (查 HF 文件列表:transformer 16 项 / transformer_ref 16 项 / text_encoder 23 项 / vae 5 项,
> 其中 46 个是 `.safetensors`),只排两个子树会额外拉下几百 GB。本机磁盘上这几个顶层目录
> **只有 config/index/tokenizer 类小文件**(`transformer` 2 个文件 72K、`text_encoder` 8 个文件 9.5M),
> 正是因为当初逐个列了文件名。所以 `--exclude "*.safetensors"` 这一条是**必须的**。
> `--exclude` 可重复给(CLI 里是 `list[str]`),`--dry-run` 可以先干看清单不落盘。

**权重来源与访问权限**(实测 HF API `gated` / `private` 字段,采集日 2026-08-09)：

| repo | gated | private | 认证 | 取法 |
|---|---|---|---|---|
| `MiniMaxAI/MiniMax-H3` | false | false | 建议带 token | `hf download`(匿名会被限速,见 §8 第 5 条) |
| `larryvrh/MiniMax-H3-Turbo-Lora` | false | false | 建议带 token | 同上 |
| `Comfy-Org/MiniMax-H3` | false | false | **不需要** | `download_h3.sh:8,36` / `download_bf16.sh` 走**裸 `curl -sSL --fail -C -`,全程无 Authorization 头** |

> 三个 repo 当前**都不是 gated**,所以 401/403 基本只会是「token 没写对 / 限速」而不是「没过门禁」。
> 但上游随时可能改成 gated —— 若真遇到 403,先去网页看是否需要 accept license,再回来重跑 `hf auth login`。

```bash
# (e) HF cache 桥接 —— 让 SGLang 用仓库名也能命中本地文件,免于重下 145G
SNAP=~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/101ecd0aa25532b6443c625340f095a617a2526a
mkdir -p "$SNAP" ~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/refs
# 把 $STORE/MiniMax-H3 下每一项逐个 symlink 进 $SNAP(目录建实体目录,文件建 symlink)
echo -n 101ecd0aa25532b6443c625340f095a617a2526a > ~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/refs/main

# (f) 收尾:清掉 hf download 的暂存残留(本机至今还留着 11G 没清,见 §8 第 1 条)
find "$STORE/MiniMax-H3/.cache" -name "*.incomplete" -printf "%s %p\n" | sort -n   # 先看,确认没有下载在跑
# 确认无误后再删: find "$STORE/MiniMax-H3/.cache" -name "*.incomplete" -delete
```
> ⚠️ **本机的桥接写的是 `b3c7290…`，与磁盘真实 revision `101ecd0…` 不一致**(见 §5.2)。
> 复现时**用真实 revision**，别抄这个不一致。

### 步骤 7 — 合并 Turbo LoRA

```bash
$VLM_PY "$BASE/scripts/merge_turbo_lora.py" \
  --lora   "$STORE/loras/minimax_h3_turbo_v4_step600_ema.safetensors" \
  --lora-revision afc0346516372a17162c14df3c5264de1d9aa1c0 \
  --base   "$STORE/MiniMax-H3/FL2VA" \
  --dst    "$STORE/MiniMax-H3-Turbo-v4s600ema" \
  --strength 1.0 \
  --comfy-single "$BASE/ComfyUI/models/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
```
**为什么 strength 1.0 / alpha 缺省**：LoRA 文件里没有 alpha 元数据，约定 `alpha = rank`(scale 1.0)。
**为什么给 `--comfy-single`**：这台机器同时有 HF 分片和 Comfy 单文件，能跑 **L2 独立布局 oracle**
(qkv 的 group-interleaved 磁盘布局 ↔ Q_all/K_all/V_all 运行时布局的重排是最容易出错的一环)。
其他机器没有单文件，才用 `--comfy-single /nonexistent` 跳过 L2 并靠溯源哈希比对替代。

期望日志(本机实测值)：
```
LoRA: 518 tensors -> 259 pairs; alpha metadata absent -> convention alpha=rank (scale=1.0)
verify L1 passed: 259 modified + 276 unmodified = 535 tensors bit-checked
verify L2 passed: Comfy single-file == reorder(HF disk) bit-exact on 6 qkv + 5 plain layers
verify L3 passed: worst min_row_cos=0.999999, worst relL2=9.65e-04
DONE -> …/MiniMax-H3-Turbo-v4s600ema (manifest + .complete written)
```
```bash
# SGLang 专用别名目录:native H3 识别按路径 basename 短名匹配
mkdir -p "$STORE/turbo_v4s600ema_alias"
ln -s "$STORE/MiniMax-H3-Turbo-v4s600ema" "$STORE/turbo_v4s600ema_alias/MiniMax-H3"
```
> ⚠️ 别名的 **leaf 必须叫 `MiniMax-H3`**。SGLang 用
> `KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS` 按路径 basename 短名 == `minimax-h3` 匹配；
> 根 `model_index.json` 的 `_class_name` 是 `MiniMaxH3ModularPipeline`，registry 不认，
> 会跌回 diffusers 路径然后崩。`h3_switch.sh` 会自动建这个 symlink。
> 另外 merged 根**必须是完整 HF 布局**(把 base 根的所有顶层项 symlink 过来)，不能只有 `FL2VA/`。

### 步骤 8 — 冒烟验证

```bash
# 8.1 三个 env 的 torch 都能看到 4 张卡
for E in h3_comfy_NV_py312 h3_vllm_NV_py312 h3_sglang_NV_py312; do
  $HOME/miniconda3/envs/$E/bin/python -c \
    "import torch;print('$E',torch.__version__,torch.version.cuda,torch.cuda.device_count())"
done
# 期望(三行一模一样):
#   h3_comfy_NV_py312  2.11.0+cu130 13.0 4
#   h3_vllm_NV_py312   2.11.0+cu130 13.0 4
#   h3_sglang_NV_py312 2.11.0+cu130 13.0 4
# 注意:sglang env 的 `pip show torch` / `pip freeze` 只记 `2.11.0`(uv 装时丢了 local version),
#      但 torch.__version__ 与另两个 env 一致带 +cu130,是同一个 cu130 wheel(git_version 相同)。
#      差异只在 pip 元数据层面,不要因此重装 torch。

# 8.2 后端可 import 且带 H3
$VLM_PY -c "import vllm,vllm_omni,pathlib;print(vllm.__version__, vllm_omni.__file__); \
  print('minimax_h3:',(pathlib.Path(vllm_omni.__file__).parent/'diffusion/models/minimax_h3').exists())"
# 期望: 0.26.0  …/src/vllm-omni/vllm_omni/__init__.py  minimax_h3: True
#      (伴随一条 version-mismatch RuntimeWarning,属正常)
$SGL_PY -c "import sglang;from sglang.multimodal_gen.configs.pipeline_configs import minimax_h3;print(sglang.__version__,'OK')"
# 期望: 0.0.0.dev1+g407a65d3c OK   <- 版本串是安装时快照,以 git 为准

# 8.3 补丁三连
git -C "$SRC/sglang" rev-parse --abbrev-ref HEAD          # turbo-lora-backport
sed -n '/class PinnedResidentLayerGroup/,/def offload/p' \
  "$SRC/vllm-omni-pr5910/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py" | grep -c as_strided   # >=1
grep -n "_ASYNC_OUTPUT_TIMEOUT = " "$SRC/vllm-omni/vllm_omni/diffusion/diffusion_engine.py"                      # 期望已改大

# 8.4 merged 权重契约
test -f "$STORE/MiniMax-H3-Turbo-v4s600ema/.complete" && echo COMPLETE_OK
python3 -c "import json;m=json.load(open('$STORE/MiniMax-H3-Turbo-v4s600ema/merge_manifest.json'));\
print(m['lora_sha256'],m['modified_tensor_count'],m['total_tensor_count'])"
# 期望: 5f3a626c… 259 535

# 8.5 端到端(从 Mac 提交,会自动切后端 + 等健康 + 回传 mp4)
#   scripts/h3_generate.py --host 6000a --profile vllm-fp8-turbo-tp4 \
#     --first a.png --last b.png --prompt-file p.txt --name smoke
# 期望产物: 124 帧 / 5.167s / 24fps / AAC 立体声 32kHz
```
> ⚠️ **`/health` 在两个引擎上都早于权重加载就绪** → 切换后第一单必然是 warmup，**不计时**。
> ⚠️ 帧数必须 `n % 17 == 5`(124 帧)，画布必须 32 的倍数。
> ⚠️ 步数语义：ComfyUI 是 `steps == forwards`；vLLM/SGLang 是 `num_inference_steps = NFE + 1`
> (引擎 pin `vllm-omni a874b8e09` / `sglang 407a65d3` 下实测)。两边由客户端 `--nfe` 统一。
> **升级引擎后必须重跑 `scripts/test_h3_schedule.py` 复核该语义**(上游有开放 PR 要改)。

---

## 8. 已知坑与注意事项

> 本节混了两类来源，逐条标注：**[实测]** = 本次 2026-08-09 采集直接观察到；
> **[档案]** = 沿用 `claude_history/` 的历史结论，本次**未复现**(多数需要跑生成任务，与只读约束冲突)。

**磁盘 / 权重**

1. **11 GiB 残留 `.incomplete`**(实测)：`/home/isaac/Data/h3_weights/MiniMax-H3/.cache/huggingface/download/FL2VA/text_encoder/`
   下有 **8 个 0.4–2.0 GiB 的 `*.incomplete` 文件**(最小 409,509,409 B、最大 2,006,370,001 B;
   时间戳 2026-08-06 00:10–00:14)，合计 `du -sh .cache` = **11 GiB**，是 145G 下载期间被 kill/重启留下的。
   这正好解释了 `du MiniMax-H3` = 145G 而 `FL2VA` 子树只有 135G。**两块盘都已 93–94% 满，这 11G 可直接回收**
   (删除前请确认没有下载在跑)。
2. **根盘 94% 满**且 `ComfyUI/models` 的 51G 实体在根盘、又落在 Dropbox 同步目录里。新增权重一律往
   `/home/isaac/Data/h3_weights` 放并 symlink 回去(`download_bf16.sh` 已经这么做，`download_h3.sh` 没有)。
3. **HF revision 不自洽**：cache 桥接声明 `b3c7290…`，磁盘实际是 `101ecd0…`(§5.2)。溯源以后者为准。
4. **`HF_HUB_ENABLE_HF_TRANSFER` 已被 huggingface_hub 弃用**：本机 `logs/download_toplevel.log` 里明确有
   `FutureWarning: … deprecated as 'hf_transfer' is not used anymore. Please use HF_XET_HIGH_PERFORMANCE instead`。
   `scripts/dl_fl2va_auth.sh` **仍在用旧变量** —— 那次 FL2VA 下载耗时 **4 小时 35 分**(`logs/download_hf_fl2va.log`)。
   下次改用 `HF_XET_HIGH_PERFORMANCE=1`。
5. 匿名下载会被 HF 限速(日流量约 300G)，必须先写 token 到 `~/.cache/huggingface/token`。
   实测三个 repo(`MiniMaxAI/MiniMax-H3`、`larryvrh/MiniMax-H3-Turbo-Lora`、`Comfy-Org/MiniMax-H3`)
   当前 `gated=false` / `private=false`，所以 401/403 一般是 token 问题或限速，不是门禁。
6. LoRA 仓整仓 22 个文件，必须 `--include` 指定单文件。
   - 同一条:**`huggingface-cli` 已被移除**(实测)。本机 `huggingface_hub 1.26.0` 下执行只打
     `Warning: huggingface-cli is deprecated and no longer works. Use hf instead.` 然后拒绝。
     登录一律用 **`hf auth login`**;`pip install "huggingface_hub[cli]"` 装出来的命令名就是 `hf`。
     `hf download` 的 `--exclude` 可重复给,`--dry-run` 可先干看清单。

**补丁 / 版本**

7. **补丁状态一律按代码内容判定**：`git diff` 会因为补丁已 commit 而误报"没打"；
   stride 补丁的 grep **必须限定在 `PinnedResidentLayerGroup` 类内**，否则恒真。
8. **`pip show` 的版本号对 editable 安装不可信**(sglang 停在 `g407a65d3c`，实际 HEAD `4c28e24`)。
   `pip freeze` 记的 commit 才是当前的。
9. **`_ASYNC_OUTPUT_TIMEOUT = 30.0` 在本机两份 checkout 里都还没改**(§4.2)。
   "500 + 空 message" 先怀疑它，别从客户端错误反推硬件。
10. `import vllm_omni` 的 version-mismatch RuntimeWarning 是噪音，不要"修"。
11. 下次升级 sglang **必须消化 `turbo-lora-backport` 分支**，否则 dynamic LoRA 路线再次 IndexError。

**运行 / 拓扑**

12. **SGLang 公共 API 硬锁 `short_edge=768`**(源码实证)，低延迟小画布在这个 commit 上走不通。
13. **768P + TP4 的 BF16 在 48G 卡 OOM** → SGLang 产线固化 `--quantization fp8`。
14. **SGLang 4 rank 并发 CPU 暂存会打爆 RAM** → `RUNAI_STREAMER_MEMORY_LIMIT=8589934592`。
15. **vLLM-Omni TP4 的 runtime LoRA 对 fused 层只 warning 后跳过**(qkv 21504 行 / fc1 28672 行与
    `sum(output_slices)` 对不上)→ 蒸馏**静默失效**。**vLLM 只走 merged checkpoint 路线**，
    dynamic LoRA 只在 SGLang 上做(且必须 `--lora-merge-mode dynamic`，`auto` 会把 LoRA merge 进 FP8 权重)。
16. 三条 serving 产线**互斥占 4 卡**；`h3_switch.sh` 用 `flock` 串行化，守护进程启动处都带 `9>&-`
    关闭继承的锁 fd(不加会导致守护进程活着就一直持锁，后续切换干等 600s)。
17. **`h3_switch.sh` 只写 `run/vllm.variant`,不写 `run/vllm.model`**(实测 `run/` 下只有 `vllm.pid`、
    `vllm.variant`、`.switch.lock`)。5090 的 switcher 两个都写。6000a 因为 `vllm` 与 `vllm-turbo`
    走不同 `model_root` 而变体标签里已含 `turbo-` 前缀，**当前不会服错 checkpoint**，
    但若以后加同 root 不同精度的变体，要补上 `.model` 追踪。
18. **DisTorch donor 未落 gpu3**(DiT 溢出走内存流式)：正确性无损，只是慢；未精调。
19. ComfyUI 会缓存整图，同参数重跑要变 seed/prompt，否则计时假快。
20. benchmark 数字**必须声明冷热态**；`/health` 早于权重就绪。

**协作**

21. 这台机器是 isaac 的日常工作机，GPU 上可能有别人的活(采集时 GPU2 上有 gnome-shell / warp-terminal / TeamViewer)。
    历史上 isaac 的 kreaid ComfyUI 占过 GPU0/1(:8188/:8189)，**经用户批准才停用**，恢复命令在 `/home/isaac/workdir/kreaid/`。
    动服务前先看 `ss -ltnp` 和 `nvidia-smi`。
22. ssh 时**远端路径一律用单引号或 `\$HOME`**，双引号里的 `~` 会被 Mac 本地 shell 展开成 `/Users/ning`。
    `pgrep`/`pkill` 的模式用方括号(`'[v]llm serve'`)防自匹配 —— **wait 循环同样适用**。

---

## 9. 本文档的采集方式

- **采集日期**：2026-08-09(远端时钟 21:31–21:45 AEST)。全程**只读**，未在 popos-6000a 上写入或修改任何文件。
- **原始清单落盘**(Mac 仓库)：
  - `doc/machines/locks/popos-6000a.h3_comfy_NV_py312.pip-freeze.txt` / `.conda-env.yml`
  - `doc/machines/locks/popos-6000a.h3_vllm_NV_py312.pip-freeze.txt` / `.conda-env.yml`
  - `doc/machines/locks/popos-6000a.h3_sglang_NV_py312.pip-freeze.txt` / `.conda-env.yml`

  每份开头有采集时间/机器/环境路径注释。已按
  `(hf|ghp|gho|ghs|github_pat)_[A-Za-z0-9]{16,}` / `sk-or-v1-` / `sk-ant-` / `AKIA` /
  `://user:pass@` 正则扫过，**0 命中**(freeze 里的两条 `-e git+https://…` 是干净的公开 URL，不含 token)。
- **重新采集**：三条 ssh 批处理即可 ——
  ① 硬件系统(`nvidia-smi` / `topo -m` / `topo -p2p r,w` / `lscpu` / `free -g` / `df -h` / `/etc/os-release` / `uname -r` / `gcc --version`)；
  ② Python 环境(`conda env list`；对三个 `h3_*` env 跑 `python -V`、`pip --version`、`pip freeze`、
  `conda env export`、`pip show <关键包>`、`python -c "import torch;…"`)；
  ③ 源码与权重(对四个 checkout 跑 `git remote -v/rev-parse/status --short/log -1`；
  补丁用 `sed -n '/class PinnedResidentLayerGroup/,/def offload/p' … | grep as_strided` 与
  `grep -n B.dim\(\)\ ==\ 3 …/lora/linear.py` 与 `grep -n _ASYNC_OUTPUT_TIMEOUT …`；
  权重用 `du -sh`、`ls -la`、`cat merge_manifest.json`、`sha256sum loras/*`、
  `find …/.cache/huggingface/download -name "*.metadata" -exec head -1 {} \; | sort | uniq -c`)。
- **非 ssh 来源(已在正文标注)**：`https://huggingface.co/api/models/<repo>`(取 `gated`/`private`)与
  `…/api/models/MiniMaxAI/MiniMax-H3/revision/101ecd0…`(取该 revision 的完整文件列表,用来验证
  §7 步骤 6(b) 的 `--exclude` 组合会留下哪些文件)。这两条是**公开只读 API**,不带任何凭据。
- **交叉对照的仓库内素材**：`scripts/h3_generate.py`(HOSTS / LEGACY_ALIASES / `--list`)、
  `scripts/h3_switch.sh`、`scripts/launch_comfy_6000a_4gpu.sh`、`scripts/setup_h3.sh`、
  `scripts/install_backends.sh` / `install_backends2.sh`、`scripts/download_h3.sh` / `download_bf16.sh` / `dl_fl2va_auth.sh`、
  `scripts/merge_turbo_lora.py` / `check_turbo_lora.py`，以及远端 `logs/` 下的
  `install_vllm_src.log`、`install_sglang3.log`、`install_backends2.log`、`merge_turbo_lora.log`、
  `download_hf_fl2va.log`、`download_toplevel.log`、`lora_download.log`、`vllm_server.log`。
