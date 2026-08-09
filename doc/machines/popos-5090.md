# popos-5090 机器档案

> 采集日期 **2026-08-09**(机器本地时区 +10:00)。本文所有事实均来自当日在该机上实际执行的命令；
> 凡未实测的内容都显式标注为「推测」并给出依据。与历史档案冲突之处已在正文中点名。
> 重新采集方式见第 9 节。

---

## 1. 这台机器是什么

**双卡 RTX 5090 的本地工作站,是 H3 项目里唯一同时跑「ComfyUI 单卡量化产线」和「vLLM-Omni TP2 serving 产线」的机器,也是 stride 补丁的根因定位现场。**

| 项 | 实测值 | 备注 |
|---|---|---|
| ssh 别名 | `popos-5090` | |
| hostname | `Ubuntu-5090` | ⚠️ 见下方「命名冲突」 |
| 项目根 | `/home/isaac/data/dropbox/CV/h3` | = git 工作区,`origin` 为 `https://github.com/stargazerscripter123/H3_inference.git`,采集时 `main @ 39c4b5089f78f25ae62b5df19fa2e5f9492768c8`,worktree 干净 |
| GPU | 2 × NVIDIA GeForce RTX 5090,各 **32607 MiB**,compute capability **12.0(sm_120)** | GPU0 `0000:21:00.0`(UUID `GPU-1762af7b-…`)/ GPU1 `0000:C3:00.0`(UUID `GPU-d7b88726-…`) |
| GPU 互联 | `nvidia-smi topo -m` → **NODE**(同 NUMA 内跨 PCIe Host Bridge) | 不是 SYS |
| P2P | `topo -p2p r` 与 `-p2p w` 双向 **全 CNS(Chipset not supported)** | **无 P2P**,所有 NCCL 走 host 中转 |
| CPU | AMD **Ryzen Threadripper 9960X**,24 核 / 48 线程,单 socket | max 5489.76 MHz,L3 128 MiB(4 instances) |
| NUMA | **1 个 node**(node0 = CPU 0-47) | 与 runpods(2 NUMA)不同,TP 分组无 NUMA 约束 |
| 内存 | `free -g` total **125 GiB**(ComfyUI 报 128149 MB),swap 7 GiB | 采集瞬间 used 49 / available 75 |
| 磁盘(H3 用的那块) | 1.9T NVMe(`nvme1n1p3` → LUKS `dm_crypt-0` → LVM `ubuntu--vg-ubuntu--lv`)挂 `/`,**1.9T 总容量,已用 475G,可用 1.3T(27%)** | H3 的全部权重/源码/ComfyUI 都在这块盘上(`stat -c %m ~/data/dropbox/CV/h3` → `/`,`~/data` 是真实目录不是 symlink) |
| 其他盘(与 H3 无关) | `nvme0n1` 3.6T LUKS → **`/home/isaac/Data`**(2.5T 已用 / 1007G 可用,里面是另一份 `Dropbox/`);`nvme2n1` 1.8T LUKS → `/home/isaac/workdir`(他人 kreaid 工作区);`sda` 4.5T LUKS → `/media/isaac/Backup` | **不要动**。⚠️ 注意 `/home/isaac/Data`(大写 D)与项目根 `/home/isaac/data`(小写 d)是**两个不同的东西** |

### 命名冲突(与历史档案不符,以实测为准)

- ssh 别名叫 `popos-5090`,但 `/etc/os-release` 是 **`Ubuntu 24.04.4 LTS (Noble Numbat)`**,hostname `Ubuntu-5090`。
  **这台机器不是 Pop!\_OS**。别名是历史遗留,写脚本/文档时不要据别名推断发行版。

### 磁盘布局与 6000a 的关键差异

历史档案里 6000a 因根盘 92-93% 满而把大文件放到第二块 NVMe(`/home/isaac/Data/h3_weights/`)并 symlink 回 `ComfyUI/models`。
**5090 上没有这套结构** —— 根盘还有 1.3T 可用,`ComfyUI/models/*` 里全是真实文件(唯一例外是 `models/loras/` 下那个 lora symlink,见第 5 节)。

> ⚠️ **这里有个容易踩的陷阱**:5090 上 `/home/isaac/Data` **确实存在**(3.6T LUKS 盘,1007G 可用),
> 只是 `Data/h3_weights/` 这个子目录不存在。所以 `scripts/download_bf16.sh`(`STORE="/home/isaac/Data/h3_weights"` 写死)
> 在 5090 上跑**不会因为路径不存在而失败** —— 它会 `mkdir -p` 出来,然后把 62G + 51.5G 写进那块**属于另一份 Dropbox 的盘**,
> 再 symlink 回 `ComfyUI/models`。那是 6000a 的布局,不是这台机器的布局。**5090 上一律不要用 `download_bf16.sh`**,
> 需要 BF16 单文件时按第 7 节步骤 8.6 直接 `curl` 到 `ComfyUI/models/diffusion_models/` 下。
> (该脚本还有两个 5090 上不成立的行为:开头 `while pgrep -f download_h3.sh; do sleep 60; done` 会一直等一个不存在的进程;
> 它同时还下一份 51.5G 的 `qwen3vl_32b_minimax_h3_bf16.safetensors`,而 5090 的产线根本不用它。)

### 它承担哪些 profile

`scripts/h3_generate.py --list` 中 5090 那几行(实跑输出):

| profile | 后端 | 默认步数 | 画布 |
|---|---|---|---|
| `comfy-int8-original-1c` | comfy:gpu1:8188 | 30 步 | 832×480 |
| `comfy-int8-teacache-1c` | comfy:gpu0:8189 | 12 步 | 864×480 |
| `comfy-int8-turbo-1c` | comfy:gpu0:8190 | NFE 6 | 864×480 |
| `vllm-fp8-original-tp2` | serving:vllm(:8091) | NFE 11 | 864×480 |
| `vllm-fp8-turbo-tp2` | serving:vllm-turbo(:8091) | NFE 6 | 864×480 |

旧名别名(按机器解析):`baseline`→`comfy-int8-original-1c`、`turbo`→`comfy-int8-teacache-1c`、
`turbo-lora`/`vllm-turbo`→`vllm-fp8-turbo-tp2`、`vllm`→`vllm-fp8-original-tp2`。

> ⚠️ **GPU0 是显示卡**(实测 `nvidia-smi --query-gpu=display_active` → GPU0 `Enabled`,GPU1 `Disabled`)。
> 采集时 GPU0 上常驻 Xorg 248 MiB + gnome-shell 51 MiB + TeamViewer 32 MiB + warp-terminal 214 MiB ≈ **0.54 GiB**。
> 而 :8189 / :8190 两个 ComfyUI worker 恰恰跑在 GPU0(`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`)。
> 32.6 GiB 的卡在 int8 产线本来就紧,桌面一开大窗口就可能压线。**历史档案完全没记这件事。**
> 需要极限显存时优先用 GPU1(`launch_comfy.sh 1 8188`)。

**采集瞬间的机器状态**(供理解上下文,非稳态):只有一个 ComfyUI Turbo-LoRA worker 在 `:8190` 运行
(pid 1836258,GPU0 占 26636 MiB);`:8188`/`:8189`/`:8091`/`:30010` 全部空闲;`run/` 目录里只有
`.switch.lock`,没有任何 `*.pid` / `*.variant`。

---

## 2. 系统与驱动

| 项 | 实测值 | 命令 |
|---|---|---|
| 发行版 | Ubuntu 24.04.4 LTS (noble) | `cat /etc/os-release` |
| 内核 | **`7.0.0-28-generic`**(`#28~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Wed Jul 1 15:50:57 UTC 2026`) | `uname -a` |
| NVIDIA 驱动 | **`580.173.02`** | `nvidia-smi` |
| 驱动暴露的 CUDA | **13.0** | `nvidia-smi` 表头 |
| gcc | `13.3.0`(Ubuntu 13.3.0-6ubuntu2~24.04.1),`/usr/bin/gcc` | `gcc --version` |
| 系统 nvcc | **没有**(未装系统级 CUDA toolkit) | `nvcc --version` 无输出 |
| conda | `26.7.0`,`/home/isaac/miniconda3`(base 是 py3.13) | `conda --version` |

### 驱动版本决定了哪些 wheel 变体(这一条是复现的关键)

- CUDA **13.0** 运行时要求驱动 **≥ 580.65**;本机 **580.173.02 满足**。
  → 所以这台机器可以直接用 **PyPI / `download.pytorch.org/whl/cu130` 的 `torch==2.11.0+cu130`**,
    也可以直接 `pip install vllm==0.26.0`(PyPI wheel 会拉 `torch 2.11.0+cu130`)。
- 对照:runpods 那台驱动 `570.195.03`(只到 CUDA 12.8),同样的 PyPI 安装会在启动时抛
  `RuntimeError: The NVIDIA driver on your system is too old (found version 12080)`,必须换 `+cu129` wheel。
  **这套 `+cu129` 绕行方案在 5090 上不需要,别照抄 runpods 的 `setup_runpods.sh` / `fix_cu129.sh`。**
- 编译类依赖(SGLang JIT)不依赖系统 nvcc,而是用 pip wheel 里的
  `.../site-packages/nvidia/cu13/bin/nvcc`(实测版本 **CUDA 13.4, V13.4.46**)。

### sm_120 支持

三个环境里的 torch 都实测 `torch.cuda.get_arch_list()` 含 `sm_120`:
`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`,`torch.cuda.device_count() == 2`。

---

## 3. Python 环境

**没有 venv**;全部走 conda(`/home/isaac/miniconda3`)。conda 里一共 10 个 env,其中
**只有 3 个属于 H3**;其余(`dinov3_NV_py312` / `krea_NV_py312` / `kreaid_NV_py312` /
`locateanything_NV_py311` / `rfdetr_NV_py312` / `wan-svi` / `wan-svi_Cu130_Py312`)是别人的项目
—— **这台机器是共享的,不要动它们**。

三个 H3 环境的 python 都是 **3.12.13**,pip 都是 **26.1.2**。
conda 只管 python / openssl / libgcc 这一层,**其余全部是 pip 装的**。包数按口径分开记(三个数字互不相等,
比对时务必先对齐口径):

| 口径 | 命令 | comfy | vllm | sglang |
|---|---|---|---|---|
| pip 可见的包 | `pip freeze --all \| wc -l` | **108** | **230** | **239** |
| conda 视角全量 | `conda list -n <env> \| grep -vc '^#'` | 135 | 257 | 266 |
| 其中来自 pypi 通道 | `conda list -n <env> \| grep -c pypi` | 105 | 228 | 236 |

`conda list` 的全量比 `pip freeze` 多出的那 27/27/27 条,是 conda 自带的 python / openssl / libgcc / ca-certificates
这一层(pip 看不到)。因此复现时 **以 `.pip-freeze.txt` 为准**,`.conda-env.yml` 只作参照。

> ⚠️ 别用 `pip list | wc -l` 数包 —— 它带 2 行表头,会把每个数字系统性抬高 2(历史版本的 110/232/241 就是这么来的)。

清单落盘位置(本次采集产出;行数 = 4 行文件头注释 + `pip freeze --all` 的条目数):

```
doc/machines/locks/popos-5090.h3_comfy_NV_py312.pip-freeze.txt    (112 行 = 4 + 108)
doc/machines/locks/popos-5090.h3_comfy_NV_py312.conda-env.yml
doc/machines/locks/popos-5090.h3_vllm_NV_py312.pip-freeze.txt     (234 行 = 4 + 230)
doc/machines/locks/popos-5090.h3_vllm_NV_py312.conda-env.yml
doc/machines/locks/popos-5090.h3_sglang_NV_py312.pip-freeze.txt   (243 行 = 4 + 239)
doc/machines/locks/popos-5090.h3_sglang_NV_py312.conda-env.yml
```

已扫描确认这 6 个文件里**没有任何 token/密钥**(无 `hf_` / `ghp_` / `sk-or-v1-` 前缀的长串,无带凭据的 URL)。
唯一的两条 `git+https://` 是 editable 安装的溯源 URL,不含凭据。

### 3.0 环境 → 用途 对应关系(交叉对照三处证据)

| 环境 | 谁在用它 | 证据 |
|---|---|---|
| `h3_comfy_NV_py312` | ① 三个 ComfyUI worker(:8188/:8189/:8190);② **所有 runner / 客户端脚本**(`run_fl2va.py`、`run_fl2va_vllm.py`、`run_fl2va_sglang.py`) | `launch_comfy*.sh` 的 `ENVPY=`;`h3_generate.py:347` 的 `rpy` 默认值 `~/miniconda3/envs/h3_comfy_NV_py312/bin/python`;`h3_generate.py:473` 的 comfy runner 命令 |
| `h3_vllm_NV_py312` | **只跑 vLLM 服务端**(`$VLM_BIN/vllm serve`) | `h3_switch_5090.sh` 的 `VLM_BIN` |
| `h3_sglang_NV_py312` | **只跑 SGLang 服务端**(`$SGL_BIN/sglang serve`) | `h3_switch_5090.sh` 的 `SGL_BIN` / `_launch_sglang` |

> 注意一个容易误解的点:5090 的 `HOSTS["5090"]` **没有** `remote_python` 字段,
> 所以 `h3_generate.py` 用默认值 —— 也就是说 **vLLM/SGLang 的客户端是在 comfy 环境里跑的**。
> 这是对的,不是配置错误:三个 client 脚本都只用标准库(`argparse/json/subprocess/urllib/hashlib/...`),
> 不 import torch/vllm/sglang。只有 runpods 因为没有 conda 才显式覆盖 `remote_python`。

---

### 3.1 `h3_comfy_NV_py312` —— ComfyUI 产线 + 全部客户端脚本

- 路径:`/home/isaac/miniconda3/envs/h3_comfy_NV_py312`
- Python 3.12.13 / pip 26.1.2 / **108 条 `pip freeze --all`**(conda list 全量 135,其中 105 来自 pypi)
- `torch.__version__ = 2.11.0+cu130`,`pip show torch` 的 Version 也是 `2.11.0+cu130`,`torch.version.cuda = 13.0`,`device_count() = 2`

| 包 | 版本 | 安装形态 |
|---|---|---|
| torch | **2.11.0+cu130** | wheel,site-packages |
| torchvision | **0.26.0+cu130** | wheel |
| torchaudio | **2.11.0+cu130** | wheel |
| torchsde | 0.2.6 | wheel |
| triton | 3.6.0 | wheel |
| **comfy-kitchen** | **0.2.26** | wheel(量化算子,必装) |
| **comfy-aimdo** | **0.4.11** | wheel(必装) |
| comfy-angle | 0.1.0 | wheel |
| transformers | 5.14.1 | wheel |
| safetensors | 0.8.0 | wheel |
| numpy | **2.4.4** | wheel(注意:比另外两个环境新) |
| huggingface_hub | 1.26.0 | wheel |
| hf-xet / hf-transfer | 1.6.0 / 0.1.9 | wheel |
| modelscope / modelscope-hub | 1.39.1 / 0.2.0 | wheel(TE FP8 换源下载用) |
| einops | 0.8.2 | wheel |
| nvidia-cuda-runtime | 13.0.96 | wheel |
| nvidia-cudnn-cu13 | 9.19.0.56 | wheel |
| nvidia-nccl-cu13 | 2.28.9 | wheel |

- **没有 editable 安装**(`pip show` 里没有任何 `Editable project location`)。ComfyUI 本身不是 pip 包,是源码目录直接 `python main.py`。
- **没有 flash-attn / sageattention / xformers / diffusers / accelerate**。
  → profile 命名规范里的 `sage` / `flash` 变体在这台机器上**目前没有物质基础**。
- `site-packages/nvidia/cu13/lib/` 里**已有无版本 `libcudart.so` 符号链接**(`launch_comfy_5090.sh` 的
  `LD_LIBRARY_PATH` 就指向这里,是当年为 ComfyUI-MultiGPU 的 `CDLL("libcudart.so")` 探测做的)。

### 3.2 `h3_vllm_NV_py312` —— vLLM-Omni TP2 serving(生产)

- 路径:`/home/isaac/miniconda3/envs/h3_vllm_NV_py312`
- Python 3.12.13 / pip 26.1.2 / **230 条 `pip freeze --all`**(conda list 全量 257,其中 228 来自 pypi)
- `torch 2.11.0+cu130`(`pip show torch` 同为 `2.11.0+cu130`),`torch.version.cuda 13.0`,`device_count 2`

| 包 | 版本 | 安装形态 |
|---|---|---|
| **vllm** | **0.26.0** | wheel,site-packages |
| **vllm-omni** | **0.26.0** | **editable →** `/home/isaac/data/dropbox/CV/h3/src/vllm-omni-pr5910` |
| torch / torchvision / torchaudio | 2.11.0+cu130 / 0.26.0+cu130 / 2.11.0+cu130 | wheel |
| torchcodec | 0.15.0+cu130 | wheel |
| triton | 3.6.0 | wheel |
| transformers | 5.14.1 | wheel |
| diffusers | **0.38.0** | wheel |
| accelerate | 1.12.0 | wheel |
| safetensors | 0.8.0 | wheel |
| numpy | **2.3.5** | wheel |
| flashinfer-python | 0.6.14 | wheel |
| humming-kernels | 0.1.10 | wheel(FP8 kernel 之一) |
| nvidia-cutlass-dsl(+ libs-base/core/cu12/cu13) | 4.6.0 | wheel(**这就是 `VLLM_DISABLED_KERNELS` 要禁的那个 Cutlass 路径的来源**) |
| nvidia-cuda-nvcc / nvidia-cuda-crt / nvidia-nvvm | 13.3.73 | wheel |
| nvidia-cuda-runtime | 13.0.96 | wheel |
| setuptools-scm | 10.2.1 | wheel(editable 构建需要) |
| aenum | 3.1.16 | wheel(**`vllm_omni/patch.py` import 它,漏装会让 `import vllm_omni` 直接炸**) |

**editable 落地方式(实测)**:
`site-packages/__editable__.vllm_omni-0.26.0.pth` → `__editable___vllm_omni_0_26_0_finder.install()`,
`python -c "import vllm_omni"` 解析到
`/home/isaac/data/dropbox/CV/h3/src/vllm-omni-pr5910/vllm_omni/__init__.py`。
pip freeze 里那一行是:

```
-e git+https://github.com/vllm-project/vllm-omni.git@070096bd6872418cd8c4e1be18ba1450f301180e#egg=vllm_omni
```

> ⚠️ **这一行不能直接拿去 `pip install`**:`070096bd` 是**只存在于本机的本地分支提交**
> (stride 补丁,见第 4 节),上游没有这个 object。复现必须「checkout 上游 `b18eeff2` → 打补丁 → 本地 commit」。

> ⚠️ 与历史档案的差异:`scripts/install_5090_backends.sh` 里 editable 装的是 `src/vllm-omni`(main),
> **但当前生产 env 已经改指向 `src/vllm-omni-pr5910`**。`src/vllm-omni` 现在是**孤儿目录,没有被任何环境引用**。

- **没有 flash-attn / sageattention / xformers**(vLLM 用的是 `--diffusion-attention-backend CUDNN_ATTN`,由 `nvidia-cudnn-frontend 1.27.0` 提供)。
- `site-packages/nvidia/cu13/lib/` 里**只有 `libcudart.so.13`,没有无版本 symlink**(与 sglang env 不同,vLLM 不需要)。

### 3.3 `h3_sglang_NV_py312` —— SGLang(路线已判死,环境仍在)

- 路径:`/home/isaac/miniconda3/envs/h3_sglang_NV_py312`
- Python 3.12.13 / pip 26.1.2 / **239 条 `pip freeze --all`**(conda list 全量 266,其中 236 来自 pypi)
- `torch.__version__ = **2.11.0+cu130**`,`torch.version.cuda = 13.0`,`device_count 2`,arch 含 sm_120 —— 运行期与另外两个环境**完全一样**。

> ⚠️ **这个环境与另外两个的唯一版本差异只在 pip 元数据里,`torch.__version__` 看不出来**(旧版本档案在这里记错过):
>
> | 口径 | comfy | vllm | sglang |
> |---|---|---|---|
> | `python -c "import torch;print(torch.__version__)"` | `2.11.0+cu130` | `2.11.0+cu130` | `2.11.0+cu130` |
> | `pip show torch \| grep ^Version` | `2.11.0+cu130` | `2.11.0+cu130` | **`2.11.0`** |
> | lock 文件里的 `torch==` | `2.11.0+cu130` | `2.11.0+cu130` | **`2.11.0`**(`locks/popos-5090.h3_sglang_NV_py312.pip-freeze.txt:213`) |
>
> 也就是说:sglang env 的 torch 来自**默认 PyPI 通道**(PyPI 的 torch 2.11.0 已是 CUDA 13 构建,只是不带本地版本标记),
> comfy/vllm 是从 `download.pytorch.org/whl/cu130` 装的。**功能等价**,判据是 `pip show` / lock 文件,
> **不是 `torch.__version__`** —— 用后者去找差异永远找不到,会误以为环境装错了。

| 包 | 版本 | 安装形态 |
|---|---|---|
| **sglang** | **0.5.18.dev27+g60b9e5149** | **editable →** `/home/isaac/data/dropbox/CV/h3/src/sglang-pr33681/python` |
| sglang-kernel | 0.4.5 | wheel |
| sgl-deep-gemm | 0.1.5.post1 | wheel |
| torch / torchvision / torchaudio | pip 元数据 **2.11.0 / 0.26.0 / 2.11.0**(无 `+cuXXX`);运行期 `__version__` 仍是 `2.11.0+cu130 / 0.26.0+cu130 / 2.11.0+cu130` | wheel(PyPI 默认通道) |
| torchao | 0.17.0 | wheel |
| torchcodec | 0.11.1 | wheel |
| triton | 3.6.0 | wheel |
| **flash-attn-4** | **4.0.0b19** | wheel(三个环境里唯一有 flash-attn 的) |
| flashinfer-python | 0.6.15.post1 | wheel |
| transformers | **5.12.1** | wheel(比另外两个环境**旧**) |
| diffusers | **0.37.0** | wheel |
| safetensors | 0.8.0 | wheel |
| numpy | 2.3.5 | wheel |
| einops | **0.9.0.dev0** | wheel(预发布) |
| nvidia-cuda-nvcc / crt / nvdisasm / nvvm | **13.4.46rc1** | wheel |
| `nvidia-cuda-nvcc-cu13` / `nvidia-cuda-runtime-cu13` | `0.0.0a0` | **占位元包**,真正的实现是上面那几个 |
| nvidia-modelopt | 0.46.0rc1 | wheel |
| runai-model-streamer | 0.16.1 | wheel(对应 `RUNAI_STREAMER_MEMORY_LIMIT`) |
| torch-memory-saver | 0.0.9.post1 | wheel |
| pydantic | **2.14.0b1** | wheel(预发布;`--prerelease=allow` 的后果) |

pip freeze 里的 editable 行:

```
-e git+https://github.com/sgl-project/sglang.git@60b9e5149930acd048936fcb212777410932f227#egg=sglang&subdirectory=python
```

**JIT 编译前提(实测已就位)**:
- `.../site-packages/nvidia/cu13/bin/nvcc` 存在,`nvcc --version` → `release 13.4, V13.4.46`
- `.../site-packages/nvidia/cu13/lib/libcudart.so` → **无版本 symlink 已建**,指向 `libcudart.so.13`
- `ninja 1.13.0` 已装

> ⚠️ **重要冲突**:历史档案说 SGLang PR33681 的 TP2 路线**已判死**(在线 FP8 需先整载 33G BF16 shard > 31.3G,加载期 OOM)。
> 但当前 env 里 editable 指向的**正是这条死路线的 checkout**(`src/sglang-pr33681`,分支 `h3-fp8-te`),
> 而 main 那份(`src/sglang`)**根本没被安装**。也就是说:现在敲 `h3_switch_5090.sh sglang` 起来的是那个已判死的构建。
> **想保留这个环境作历史证据可以,但不要指望它能出片。**

---

## 4. 第三方源码与补丁

### 4.1 checkout 清单(实测)

| 路径 | 是否 git | remote | branch | HEAD(完整) | dirty | 体积 |
|---|---|---|---|---|---|---|
| `~/data/dropbox/CV/h3/ComfyUI` | ✅ | `https://github.com/comfyanonymous/ComfyUI.git` | (detached at FETCH_HEAD) | `57500fc5bc92566a63f2046824f522cd55c335ca` | 0 | — |
| `~/data/dropbox/CV/h3/src/vllm-omni-pr5910` | ✅ | `https://github.com/vllm-project/vllm-omni.git` | **`pr5910-stride-fix`** | **`070096bd6872418cd8c4e1be18ba1450f301180e`** | 0 | 149M |
| `~/data/dropbox/CV/h3/src/vllm-omni` | ❌ **不是 git 仓库** | — | — | (见下) | n/a | 68M |
| `~/data/dropbox/CV/h3/src/sglang-pr33681` | ✅ | `https://github.com/sgl-project/sglang.git` | **`h3-fp8-te`** | **`60b9e5149930acd048936fcb212777410932f227`** | 0 | 400M |
| `~/data/dropbox/CV/h3/src/sglang` | ❌ **不是 git 仓库** | — | — | (见下) | n/a | 132M |

- ComfyUI 自报版本:`ComfyUI/comfyui_version.py` → `__version__ = "0.29.0"`;
  HEAD 提交信息 `feat: Support MiniMax-H3 (CORE-375) (#15224)`(2026-08-03)。✅ 与档案一致。
- `ComfyUI/requirements.txt` 里 **自带** `comfy-kitchen==0.2.26`(第 25 行)与 `comfy-aimdo==0.4.11`(第 26 行);
  而 torch / torchvision / torchaudio 在 **第 4 / 6 / 7 行**且**无版本约束**(第 5 行是 `torchsde`)
  —— 所以 torch 版本必须靠 `setup_h3.sh` 先装好再跑 requirements,顺序不能颠倒。

> ⚠️ **新发现(档案未记)**:`src/vllm-omni` 与 `src/sglang` **都不是 git 仓库** —— 它们是当年从 6000a
> `rsync --exclude .git` 过来的裸源码树。后果:**这两份 checkout 的上游 commit 无法从工作区恢复**。
> 只能从安装日志反推(证据强度:日志字符串,可信):
> - `logs/fix_installs.log`: `vLLM-Omni version 0.1.dev1+ga874b8e09` → **`src/vllm-omni` = vllm-omni main @ `a874b8e09`**
>   (与档案「引擎侧步数语义 vllm-omni `a874b8e09`」吻合)
> - `logs/install_pr33681.log`: `- sglang==0.0.0.dev1+g407a65d3c (from file:///…/src/sglang/python)`
>   → **`src/sglang` = sglang main @ `407a65d3c`**(与 6000a 的 main pin `407a65d` 吻合)
>
> 这两个 `0.1.dev1+` / `0.0.0.dev1+` 版本号正是档案里记过的 setuptools_scm 版本回退坑的现场证据。
> **今后 rsync 源码树一律不要 `--exclude .git`。**

#### `vllm-omni-pr5910` 里的四个本地分支(实测 `for-each-ref`)

| 分支 | commit | 说明 |
|---|---|---|
| `pr5910-stride-fix` | `070096bd` (2026-08-09 10:17 +1000) | **生产分支**,= `b18eeff2` + stride 补丁 |
| `pr5910-head` | `0bcc1a9f` (2026-08-08 15:34 UTC) | `docs: add MiniMax-H3 single-GPU FP8 validation`;`b18eeff2` 是它的父提交 |
| `h3-global-fp8-dlo` | `9f921476` (2026-08-08 01:08 UTC) | `[Diffusion][Quantization] Support MiniMax-H3 encoder FP8 with DLO` |
| `main` | `81b48e83` (2026-08-08 13:14 +0800) | 上游 main 快照 |

实测 `git merge-base --is-ancestor 9f921476 b18eeff2` → **NO**,而 `b18eeff2` 是 `pr5910-head` 的祖先 → **YES**。
说明 **PR#5910 在 8-08 当天被重写/force-push 过**,`9f921476`(档案称「计划稿」)和 `b18eeff2` 不在同一条线上。
`logs/install_pr5910.log` 首行 `CLONE_DONE 9f921476` 记录的正是第一次 clone 时的旧 head。
→ **这印证了档案「PR 未合并、行为在 head 间变化,必须钉 commit 而不是钉 PR 号」的结论。**

#### ComfyUI custom_nodes(逐个实测)

| 目录 | 上游 repo | HEAD(完整) | branch | dirty |
|---|---|---|---|---|
| `ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-TeaCache` | `https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache.git` | `4cbb50d69c73a19a5d6ec42c5aec1989d5a04b6f` | (detached) | 0 |
| `ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo` | `https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo` | `55fee864dd7b2976b1c4ce3c3d5f7968f181409f` | (detached) | 0 |
| `ComfyUI/custom_nodes/ComfyUI-MultiGPU` | `https://github.com/pollockjj/ComfyUI-MultiGPU.git` | **`b51c99a525e9607e43545ee2a8b7694c74a4775a`** | `main` | 0 |

- TeaCache / Turbo 两个 commit ✅ 与档案一致。
- **`ComfyUI-MultiGPU` 的 commit 档案明确记为「没记」,现补上:`b51c99a5`(2026-05-08,`Merge PR #199 codex/aimdo-device-fallback`)。**
- `ComfyUI-MiniMax-H3-Turbo` 目录里带 **`h3_silu_temb_grid.safetensors`(5510600 B)** —— 就是给 pruned 基座运行时注入
  time-conditioning 的那份文件,**随节点仓库一起 clone,不需要单独下载**。
- 实测日志证据(`logs/comfyui_tlora.log`)Turbo 节点确实生效:
  `208 backbone modules, 158 bypass adapters, 1 injections, 50 int8 fc2 via merge + 51 adaln injected at run time`
  —— 与档案记的「51 个 adaln」完全一致。

> ⚠️ **潜在风险(档案未记)**:`ComfyUI-MultiGPU` 虽然「跨卡路线已放弃」,但它**仍然装着,并且每个 worker 启动时都会加载并 monkey-patch 核心**:
> `Patching mm.get_torch_device, mm.text_encoder_device, mm.unet_offload_device`、
> `Patched comfy.sample.sample with runtime device guard`、
> `Applied comfy_kitchen CUDA DLPack device guard patch (P2P-aware)`。
> 也就是说 :8188/:8189/:8190 三个 worker **全都跑在被 MultiGPU 改过的 model_management 上**,
> 所有历史 benchmark 数字都是在这个前提下测的。**要么保留它(推荐,保持可比),要么移除后全部重测 —— 不要在一次实验里改变它。**

### 4.2 补丁状态表(按代码内容判定,不用 `git diff`)

| 补丁 | 改哪里 | 为什么 | 不打的后果 | 判定命令 | **本机实测** |
|---|---|---|---|---|---|
| **stride 补丁**<br>`scripts/pr5910_resident_stride_fix.patch` | `src/vllm-omni-pr5910/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py`,`PinnedResidentLayerGroup.load()` | `_shard_and_pin` 按物理(列主序)顺序打包在线-FP8 的转置权重并存下 stride;streamed 路径 `prefetch_layer` 用 `as_strided` 还原得对,常驻组却用 `.view()` 按行主序重解释 | **不报错、静默输出纯噪声**(ChannelWiseTorch / BF16-dequant 等宽松 kernel),或**崩溃**(Cutlass `scaled_mm_entry.cu:209` / Humming / cuBLASLt) | `sed -n '/class PinnedResidentLayerGroup/,/def offload/p' <file> \| grep -q as_strided` | ✅ **APPLIED**(类内第 89 行 `torch.as_strided(`);分支 `pr5910-stride-fix @ 070096bd`,worktree 干净 |
| 同上,**对照组** | `src/vllm-omni/…/distributed_layerwise_backend.py`(孤儿 main 树) | — | — | 同上 | ❌ **MISSING**(类内是 `gpu_buffer[…].view(meta["shape"])`,整个文件 `as_strided` 计数 = 0)。该树未被任何环境引用,不影响生产 |
| **`_ASYNC_OUTPUT_TIMEOUT` 放大** | `vllm_omni/diffusion/diffusion_engine.py:58` | 引擎两次 yield 之间的硬上限;超时抛 `TimeoutError` → HTTP 500 + 空 message | BF16 基座这类单步耗时接近 30s 的组合会被掐掉,且报错不说原因 | `grep -n "_ASYNC_OUTPUT_TIMEOUT" <file>` | ⚠️ **未打**:两份 checkout 都仍是 `_ASYNC_OUTPUT_TIMEOUT = 30.0`(pr5910 树 :58 定义、:326 与 :793 使用) |

#### 冲突 1:`_ASYNC_OUTPUT_TIMEOUT` 补丁在 5090 上**并没有打**

历史档案把它列为「三机通用、第二个必打的本地 patch」。**实测:5090 上没打。**
这不是马上要修的故障 —— 5090 的生产组合(`vllm-fp8-*-tp2`,warm 46.9s)历史上一直能过,
正对应档案里那条「未解矛盾:5090 ~47s 却能过 → 30 秒卡的是两次 yield 之间的间隔而非整次请求」。
**结论:把它从「必打」降级为「按需打」,并在文档里说清楚判据**(见第 8 节)。

#### 冲突 2(新发现,重要):`h3_switch_5090.sh` 的 stride 探测是**恒真**的

`scripts/h3_switch_5090.sh` 里写的是:

```bash
grep -q "as_strided" "$DLO" 2>/dev/null && STRIDE="APPLIED_as_strided"
```

这是**对整个文件**做的朴素 grep。实测:

```
$ git show 070096bd^:vllm_omni/diffusion/offloader/distributed_layerwise_backend.py | grep -c as_strided
2                     # 父提交 b18eeff2(即未打补丁的状态)本来就有 2 处
                      # 行 299 physical_view = torch.as_strided(   ← streamed 路径
                      # 行 412 torch.as_strided(                   ← streamed 路径
```

→ **即使把 stride 补丁 revert 掉,这个探测依然会打印 `stride_patch=APPLIED_as_strided`。**

历史档案只把这个缺陷记在 `scripts/setup_runpods.sh` 头上,并说 `h3_switch_*.sh` 的判据是对的 —— **这一点是错的**。
`scripts/post_setup.sh` / `post_setup2.sh` 里那种**限定在类内**的写法才是正确判据:

```bash
sed -n '/class PinnedResidentLayerGroup/,/def offload/p' "$DLO" | grep -q as_strided
```

(为什么 5090 上至今没暴露:孤儿树 `src/vllm-omni` 是更早的版本,整文件 `as_strided` 计数为 0,
所以那次对照恰好给出了正确答案 —— **纯属巧合,不能当成探测有效的证据**。)

**建议(未改动任何文件,只记录)**:把 `h3_switch_5090.sh`、`h3_switch_runpods.sh`、`setup_runpods.sh` 三处的探测统一改成类内限定版。

---

## 5. 模型权重

顶层体积(`du -sh`,采集时):

| 路径 | 体积 | 拆解 | 用途 |
|---|---|---|---|
| `~/data/dropbox/CV/h3/models_official/` | **145G** | `MiniMax-H3/FL2VA` **135G** + `MiniMax-H3/.cache/` **11G**(HF 下载残留,**可删**,见 5.4)+ `Ref2VA` 29M + 根级配置/tokenizer/processor/text_encoder 约 40M | 官方 BF16 基座,vLLM `vllm-fp8-original-tp2` + merge 源 |
| `~/data/dropbox/CV/h3/models_merged/` | **62G** | 13 个真实分片(其余组件是 symlink) | Turbo-LoRA merged,`vllm-fp8-turbo-tp2` |
| `~/data/dropbox/CV/h3/ComfyUI/models/` | **85G** | 6 个单文件(见 5.1 #1–#6) | 三条 ComfyUI 产线 |
| `~/data/dropbox/CV/h3/models_te_fp8/` | **34G** | 7 个分片 | SGLang 的 TE(**路线已死,可删**) |
| `~/data/dropbox/CV/h3/models_modelopt/` | **5.0G** | 真实文件仅约 600M,其余 **4.4G 在 `.cache/`** 里是未完成分块 | **残缺,无人使用,可删**(见 5.2) |
| `~/data/dropbox/CV/h3/loras/` | **744M** | 1 个文件 | Turbo LoRA 原始文件 |

合计约 **332 GiB**,但其中 **约 50G 是可回收的死重**(`models_te_fp8` 34G + `models_modelopt` 5G +
`models_official/.cache` 11G)。生产真正需要的是 **约 283G**(ComfyUI 85G + FL2VA 135G + merged 62G + LoRA 0.7G)。
磁盘预算见第 7 节步骤 0。(⚠️ 严格只读约束下本次未删任何东西。)

### 5.1 详表

| # | 路径 | 体积 / 字节数 | 来源(HF repo @ revision) | 被哪条产线用 | 校验契约 |
|---|---|---|---|---|---|
| 1 | `ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | **20 970 379 616 B**(实测 = 脚本期望值) | `Comfy-Org/MiniMax-H3` `resolve/main`(`scripts/download_h3.sh`) | `comfy-int8-original-1c` / `-teacache-1c` / `-turbo-1c` | 下载脚本按**精确字节数**校验 + `curl -C -` 断点续传 |
| 2 | `ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | **20 958 205 608 B** | 同上;用 `scripts/dl_fp8_auth.sh` 带 token 下(repo 实测**不是 gated**,带 token 只为规避限速) | 仅 A/B 对照(实测比 INT8 慢 20%,不进生产) | 同上 |
| 3 | `ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | **15 687 142 551 B** | `Comfy-Org/MiniMax-H3` | 5090 三条 ComfyUI 产线的默认 TE(`HOSTS["5090"]["te"]`) | 字节数 |
| 4 | `ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | **27 141 342 152 B** | 同上 | 备用(6000a 的默认 TE) | 字节数 |
| 5 | `ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors` | **5 207 808 496 B** | 同上 | 全部 ComfyUI 产线 | 字节数 |
| 6 | `ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors` | **605 254 808 B** | 同上 | 全部 ComfyUI 产线 | 字节数;**fp32 是硬性要求**(BF16 会让音量 −20dB) |
| 7 | `ComfyUI/models/loras/minimax_h3_turbo_v4_step600_ema.safetensors` | symlink → #8 | — | `comfy-int8-turbo-1c`(`--turbo-lora <文件名>`) | — |
| 8 | `loras/minimax_h3_turbo_v4_step600_ema.safetensors` | **779 849 816 B**(744M) | `larryvrh/MiniMax-H3-Turbo-Lora` @ **`afc0346`** | merge 源 + ComfyUI Turbo 节点 | **实测 sha256 = `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`**,与 merged manifest 记录**逐字符一致**。G0 体检:`scripts/check_turbo_lora.py <lora路径>`(**位置参数,不是 `--lora`**;259 对键 / 518 tensors / 全 BF16) |
| 9 | `models_official/MiniMax-H3/` | **145G** = `FL2VA/` 135G + **`.cache/` 11G**(HF 下载残留,见 5.4)+ `Ref2VA/` 29M + `tokenizer/` 11M + `text_encoder/` 9.5M + `processor/` 9.4M + 根级 json/README。**下的是整仓,不是只有 FL2VA** | `MiniMaxAI/MiniMax-H3` snapshot **`b3c7290e66afdf293bef3b9077b7a266ef421f34`**(见 #11) | `h3_switch_5090.sh vllm` 的 `MODEL_ROOT/FL2VA` | `FL2VA/transformer` 实测 **13 个 safetensors 分片** + `model.safetensors.index.json`;index 的 sha256 = `fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb`(由 merge manifest 记录)。`FL2VA/` 内部:transformer 62G / text_encoder 63G / video_vae 9.8G / audio_vae 578M |
| 10 | `models_merged/MiniMax-H3-Turbo-v4s600ema/` | **62G** | 本地 merge 产出(非下载) | `h3_switch_5090.sh vllm-turbo` | **`.complete` 在根目录**(内容 = `2026-08-09T08:48:22+1000`)+ `merge_manifest.json` + `delta_norms.csv`(260 行 = 表头 + 259 条) |
| 11 | `~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/` | 676K | 桥接层,不含实体权重 | 让 SGLang 直接命中缓存(**vLLM/ComfyUI 都不需要它**) | 两个平级目录 `refs/` 与 `snapshots/`。`refs/main` = **`b3c7290e66afdf293bef3b9077b7a266ef421f34`**(40 B)✅;`snapshots/b3c7290…/` 下**只有 3 个根级 symlink**(`model_index.json` / `modular_model_index.json` / `README.md`),其余 12 项(`FL2VA` / `Ref2VA` / `scheduler` / `vae` / `transformer` / …)是**真实目录**,目录内部再逐文件 symlink 回 `models_official/`。⚠️ 这个结构不是「顶层逐项 symlink」的循环能产生的,复现命令见步骤 8.4 |
| 12 | `models_te_fp8/Qwen3-VL-32B-Instruct-FP8/` | **34G**,7 个分片(`model-0000{1..7}-of-00007.safetensors`) | `Qwen/Qwen3-VL-32B-Instruct-FP8`;HF cache 里 `refs/main` = `4bf2c2f39c37c0fede78bede4056e1f18cdf8109`(**只有 ref,没有 blob** → 实体是 ModelScope 下的) | `h3_switch_5090.sh sglang` 的 `--text-encoder-path`(路线已死) | 无字节校验脚本;`logs/download_te_ms.log` 末尾 `MS_DL_EXIT=0` + `Snapshot ready at …` |
| 13 | `models_modelopt/MiniMax-H3-FP8/` | **5.0G,残缺** | `feizhai123/MiniMax-H3-ModelOpt-Mixed9-Dynamic-FP8` @ **`80a8efc5bb9473f12ec1f0e5a1b20be5c7765fe7`**(HF cache `refs/main`) | **无人使用**(没有任何脚本引用) | 见下 |

### 5.2 `models_modelopt` —— 档案里完全没有的一份残缺权重

- HF cache 里有 `models--feizhai123--MiniMax-H3-ModelOpt-Mixed9-Dynamic-FP8`,`refs/main = 80a8efc5bb9473f12ec1f0e5a1b20be5c7765fe7`。
- 目录内容实测:`transformer/` **只有 `config.json`**;`text_encoder/` 只有
  `config.json` / `model.safetensors.index.json` / `hf_quant_config.json` / `generation_config.json` /
  `text_encoder_sensitivity_ranking.tsv` / `text_encoder_mixed_precision_config.json`;
  真正下到的大文件只有 `audio_vae/model.safetensors`(578M)、`processor/`(11M)、`tokenizer/`(11M)。
- `logs/download_modelopt.log` 显示进度停在 **38/74 files**(2026-08-08 16:01 后无更新)。
- **5.0G 里只有约 600M 是落地文件**(`audio_vae/` 578M + `processor/` 11M + `tokenizer/` 11M + 几百 KB 的 json),
  剩下的 **4.4G 全在 `models_modelopt/MiniMax-H3-FP8/.cache/`** 里,是 xet 的未完成分块。删的时候别忘了它。
- `hf_quant_config.json` 说明它是 `modelopt 0.45.0rc1.dev28+gc88b62bee` 产出的
  `FP8_PER_CHANNEL_PER_TOKEN` 混合精度权重(TE 有一串 `exclude_modules`,如 layers.17-22 的 `mlp.down_proj`)。
- **结论:一次没跑完的探索性下载,当前没有任何 profile 引用它。要么补齐(约需再下 ~60G),要么删掉回收 5G。**
  (⚠️ 严格只读约束下本次未删。)

### 5.3 merged Turbo checkpoint 的完整契约(逐字段实测)

`models_merged/MiniMax-H3-Turbo-v4s600ema/merge_manifest.json`:

| 字段 | 值 |
|---|---|
| `base_model` | `MiniMaxAI/MiniMax-H3` |
| `base_path` | `/home/isaac/data/dropbox/CV/h3/models_official/MiniMax-H3/FL2VA` |
| `base_transformer_index_sha256` | `fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb` |
| `lora_repo` / `lora_revision` / `lora_file` | `larryvrh/MiniMax-H3-Turbo-Lora` / `afc0346` / `minimax_h3_turbo_v4_step600_ema.safetensors` |
| `lora_sha256` | `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`(本次**实测重算一致**) |
| `strength` / `alpha_convention` | `1.0` / `absent -> alpha=rank (scale 1.0)` |
| `merge_dtype` / `output_dtype` | `float32` / `bfloat16` |
| `qkv_disk_layout` | `group-interleaved (56 groups x [q\|k\|v] x 128)` |
| `qkv_runtime_layout` | `Q_all_K_all_V_all (LoRA lora_B order)` |
| `merge_script_sha256` | `b034c41b1c20233333890d5c168583de48730fa9f7df10c2c4041c615b5dcb1b` ⚠️ **已失效,见下** |
| `modified_tensor_count` / `total_tensor_count` | **259 / 535** |
| `created` | `2026-08-09T08:48:22+1000` |

> ⚠️ **`merge_script_sha256` 这一项对不上,而且对不上是常态**:
> manifest 记的是**产出这份 checkpoint 时**那份脚本的哈希 `b034c41b…`,
> 但仓库当前版本(commit `39c4b508`,worktree 干净)的 `scripts/merge_turbo_lora.py` 实测
> **`70cdfe9d7b1e1dd837e61fc4fea875e383c223aaf311dd87f3f22dbe60ba777d`**(Mac 侧 `shasum -a 256` 与 5090 上 `sha256sum` 一致,
> 全树 `find -name 'merge_turbo_lora*.py'` 只有这一份;`git log -- scripts/merge_turbo_lora.py` 只有 `d66049a` 初始导入一条,
> 时间晚于 08:48 那次 merge)。**产出这份 checkpoint 的脚本已经不是今天树上的脚本了。**
>
> 后果有两条,都要记住:
>
> 1. 下面那条「靠溯源哈希比对代替 L2」的证据链里,**`merge_script_sha256` 这一环在今天的树上是断的**,
>    只剩 `base_transformer_index_sha256` / `lora_sha256` / `259` 与 `535` 计数这三项可比。
> 2. 你按第 7 节步骤 9 重跑一遍,manifest 里会写**你机器上脚本的哈希**(现在就是 `70cdfe9d…`),
>    **与本表记的 `b034c41b…` 不同,这是预期内的,不是产物损坏。**
>
> 想彻底修好:用当前脚本补跑一次 `--verify-only` 并重记本表。

目录结构实测:`FL2VA/transformer/` 是 **13 个真实文件**(权限 `-rw-------`,合计 ~62G),
`audio_vae` / `processor` / `text_encoder` / `tokenizer` / `video_vae` 全是 **symlink 指回 `models_official/.../FL2VA/`**,
`model_index.json` 是真实文件。`.complete` 与 `merge_manifest.json` 在**根目录**(不是 `FL2VA/` 里)。
(`h3_switch_5090.sh` 两个位置都接受:`[ -f "$MERGED/FL2VA/.complete" ] || [ -f "$MERGED/.complete" ]` —— 档案里那个悬而未决的问题,答案是**根目录**。)

`logs/merge_turbo.log` 记录的实际执行(**合计约 2.5 分钟** = merge ~90s + L1 校验 ~56s,
不是档案说的 ~9min,**口径以日志为准**;逐段拆解见步骤 9 的「耗时参考」与 `models.md` §3.3):

```
08:45:54 base: 535 tensors in 13 shards; heads=56 head_dim=128
08:45:54 LoRA: 518 tensors -> 259 pairs; alpha metadata absent -> convention alpha=rank (scale=1.0)
08:47:24 all 259/259 LoRA targets applied
08:47:24 top-10 |dW|_F/|W|_F: blocks.49.mlp.fc2.weight=0.0036; blocks.49.attn.qkv_proj.weight=0.0023; …
08:48:20 verify L1 passed: 259 modified + 276 unmodified = 535 tensors bit-checked
08:48:20 verify L2 SKIPPED (no --comfy-single) — layout not independently confirmed!
08:48:22 verify L3 passed: worst min_row_cos=0.999999, worst relL2=9.65e-04
```

> ⚠️ **manifest 的 `verification` 字段会骗人**:它写的是
> `"L1 bit-level recompute; L2 Comfy single-file layout oracle; L3 single-layer forward oracle …"`,
> 是一句**静态模板**,而这次 merge 的 **L2 实际被跳过了**。
> 判断 L2 是否真跑过**只能看 `logs/merge_turbo.log`,不能信 manifest**。
> 这是档案里没点破的一个坑。
>
> **为什么会跳过(机制,照抄步骤 9 的人必看)**:`scripts/merge_turbo_lora.py:264` 是
> `if args.comfy_single and os.path.exists(args.comfy_single):` —— **路径不存在就静默走 else 分支**,
> 只在日志里留一行 `verify L2 SKIPPED`,**退出码照样是 0**。本机跑的时候
> `ComfyUI/models/diffusion_models/minimax_h3_fl2va_bf16.safetensors` 根本不存在
> (实测该目录下只有 `pruned_int8_convrot` 20 970 379 616 B 与 `pruned_fp8_scaled` 20 958 205 608 B),
> 所以 L2 一次都没跑过。
>
> **本机用什么代替了 L2**:与 6000a 的溯源哈希比对 —— `base_transformer_index_sha256` /
> `lora_sha256` / `259` 与 `535` 两个计数三项一致。⚠️ 注意 **`merge_script_sha256` 那一项已经不能用了**(见上方注)。
> 这条替代链比 L2 弱:它证明「输入相同」,不证明「磁盘布局解释正确」。
> 想真正确认布局,只能按步骤 8.6 把 62 GiB 的 BF16 单文件下下来跑 L2。

### 5.4 官方基座的来源:**证据冲突,结论不确定**

- `models_official/MiniMax-H3` 的 snapshot sha 明确是 `b3c7290e66afdf293bef3b9077b7a266ef421f34`(HF cache `refs/main` 实测)。这一条是确定的。
- **来源则有两组互相打架的证据,谁也压不倒谁**:

| 指向 rsync | 指向本机 `hf download` |
|---|---|
| 本机 `logs/` 里**没有**对应的 HF 下载日志(只有 `download.log` = Comfy 单文件、`download_fp8.log`、`download_te_*.log`、`download_modelopt.log`) | `models_official/MiniMax-H3/.cache/huggingface/` 存着 **11G** 的 xet 半成品:8 个 `.incomplete` 分块(最大 **2 006 370 001 B**,全在 `download/FL2VA/text_encoder/` 下)+ 一个 `trees/`(64K) |
| 档案 `05_5090_serving` notes 明写「由 6000a rsync over LAN, 1Gbps ~25min」 | `.cache/huggingface/download/` 下的元数据目录**覆盖整个 repo**(`FL2VA` / `Ref2VA` / `scheduler` / `vae` / `transformer` / `transformer_ref` / `text_encoder` / `tokenizer` / `processor` / `audio_vae` / `audio_scheduler` / `docs` + 三个根级 `.metadata`)—— 这是 `hf download --local-dir models_official/MiniMax-H3` 才会留下的痕迹 |
| 目录 mtime 集中在 2026-08-06 00:49–04:52 | `.cache/` 的 mtime 是 **2026-08-05 23:44**,**早于**上面那个区间 |

- **→ 结论降级为「不确定」**。最能同时解释两组证据的假说是:**本机先 `hf download` 下到一半(留下 11G 半成品),
  之后改用 rsync 补齐**(所以没有一份完整的下载日志,mtime 也被 rsync 刷成了 00:49–04:52)。但这仍是推测。
- **对复现没有影响**:不管当初怎么来的,照第 7 节步骤 8.3(b) 从 HF 下即可,校验只认
  `FL2VA/transformer` 的 13 个分片 + index sha256。
- **那 11G 是纯残留,可以直接删**(`rm -rf models_official/MiniMax-H3/.cache`),
  它既不被任何产线读取,也不参与任何校验 —— 但删之前先确认没有下载正在进行。

---

## 6. 运行期环境变量

全部来自 `scripts/h3_switch_5090.sh` / `scripts/launch_comfy*.sh`(仓库文件,与机器上 git 工作区逐字节一致 —— 工作区干净)。

### 6.1 vLLM(`_launch_vllm`,profile `vllm-fp8-*-tp2`)

| 变量 | 值 | 作用 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1` | 两张卡都给引擎(TP2) |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | 多进程 worker 用 spawn,避免 fork 后 CUDA context 损坏 |
| `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` | `1800` | 视频同步 API 的整体超时(**管不到 `_ASYNC_OUTPUT_TIMEOUT` 那 30 秒**,两回事) |
| `VLLM_DISABLED_KERNELS` | `${VLLM_KERNELS_DISABLE:-CutlassFP8ScaledMMLinearKernel}` | **sm_120 上禁 Cutlass FP8 GEMM**;env 里确实装着 `nvidia-cutlass-dsl 4.6.0`,不禁就会被选中 |
| `VLLM_TEST_FORCE_FP8_MARLIN` | `${VLLM_FORCE_MARLIN:-0}` | 调试开关,默认关 |
| `VLLM_BATCH_INVARIANT` | `${VLLM_BATCH_INVARIANT:-0}` | 调试开关,默认关 |
| `VLLM_ATTN`(脚本变量,喂给 `--diffusion-attention-backend`) | 默认 `CUDNN_ATTN` | **别改成 `TORCH_SDPA`**,实测慢约 18s(46.8 → 64.8s) |
| `VLLM_DLO_RESIDENT`(脚本变量) | 默认 `50` | H3 恰好 50 层 → 全常驻 |

serve flags(同一函数):
`--omni --host 127.0.0.1 --port 8091 --trust-remote-code --num-gpus 2 --tensor-parallel-size 2
--text-encoder-tp-size 2 --usp 1 --ring 1 --quantization fp8 --enable-distributed-layerwise-offload
--dlo-no-use-allgather --dlo-resident-layers 50 --vae-patch-parallel-size 2 --vae-parallel-mode tile
--vae-use-tiling --diffusion-attention-backend CUDNN_ATTN --enforce-eager`

### 6.2 SGLang(`_launch_sglang`,路线已死但脚本仍在)

| 变量 | 值 | 作用 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1` | TP2 |
| `CUDA_HOME` | `$HOME/miniconda3/envs/h3_sglang_NV_py312/lib/python3.12/site-packages/nvidia/cu13` | JIT 找 nvcc/头文件(系统没有 CUDA toolkit) |
| `PATH` | `$SGL_BIN:$CUDA_HOME/bin:$PATH` | 让 JIT 找到 `nvcc` 与 `ninja` |
| `LIBRARY_PATH` | `$CUDA_HOME/lib` | 链接期找 `-lcudart`(靠那个**无版本 symlink**) |
| `LD_LIBRARY_PATH` | `$CUDA_HOME/lib:$LD_LIBRARY_PATH` | 运行期找 `libcudart.so.13` |
| `RUNAI_STREAMER_MEMORY_LIMIT` | `8589934592`(8 GiB) | 限制 runai streamer 的 CPU 暂存,防止多 rank 并发把 RAM 打爆 |

### 6.3 ComfyUI launcher

| 脚本 | 端口 | env |
|---|---|---|
| `launch_comfy.sh 1 8188` | 8188 | `CUDA_VISIBLE_DEVICES=1`(**没有设 `CUDA_DEVICE_ORDER`**) |
| `launch_comfy_5090_turbo.sh` | 8189 | `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`;flags `--preview-method none --async-offload 2 --reserve-vram 1.5` |
| `launch_comfy_5090_tlora.sh` | 8190 | 同上(`PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=0`),同样三个 flag |
| `launch_comfy_5090.sh`(**switch 脚本不走这条**) | 8188 | `LD_LIBRARY_PATH=…/h3_comfy_NV_py312/lib/python3.12/site-packages/nvidia/cu13/lib` + `CUDA_VISIBLE_DEVICES=1,0` 双卡可见 |

> ⚠️ **不一致(实测发现)**:`launch_comfy.sh` 没有设 `CUDA_DEVICE_ORDER`,而另外两个 launcher 设了 `PCI_BUS_ID`。
> 默认序是 `FASTEST_FIRST`。两张卡型号完全相同时通常仍等价于 PCI 序,但**这不是保证**。
> 如果哪天出现「:8188 跑到了显示卡上」的诡异现象,先查这个。目前实测 :8190 worker 确实在 GPU0(PCI `21:00.0`),与预期一致。

### 6.4 测试专用

`H3_SWITCH_LIB=1` —— `source` switch 脚本时只加载函数、不执行分派(`scripts/test_h3_switch_5090.sh` 用)。
⚠️ 它会被 export 继承:**测完不要在同一个 shell 里直接调 switcher**,否则 `TARGET` 变成 `__lib__`,切换静默变成空操作。

---

## 7. 从零复现

> 目标:在一台干净的 Ubuntu 24.04 + 2×RTX5090 + 驱动 ≥580.65 的机器上,复现本档案记录的三条 5090 产线。
> 每一步都写了**为什么**;凡是本档案或历史档案记过的坑,都用 `> ⚠️` 标在对应步骤旁。
>
> ⚠️ **这条产线是两台机器**:GPU 机(步骤 0–9、10.1–10.5、10.7)和**控制端**(步骤 10.6 / 10.8 的
> `h3_generate.py` 是在 Mac 上跑的)。控制端的前提单列在下面「前提 B」,别漏。

### 步骤 0 —— 前提 A:GPU 机(Ubuntu 24.04 + 2×RTX5090)

```bash
# 必须先确认(不确认就往下走,后面会以看不懂的方式失败)
nvidia-smi                      # 期望: Driver Version ≥ 580.65, CUDA Version 13.0
nvidia-smi topo -p2p r          # 期望: CNS —— 记住这台机器没有 P2P
free -g                         # 期望: total ≈ 125
df -h /                         # 磁盘预算见下表
gcc --version                   # 13.x 即可
```

**磁盘预算(分档,按你要复现到哪一步取)**:

| 档位 | 增量 | 累计需要 | 内容 |
|---|---|---|---|
| ① 生产最小集 | — | **≈ 285G** | ComfyUI 6 个单文件 85G + 官方 BF16 基座 `FL2VA` 135G + merged Turbo 62G + LoRA 0.75G |
| ② `hf download` 的瞬时开销 | +11G | **≈ 296G** | `models_official/MiniMax-H3/.cache/`(xet 半成品;下完可删,本机就留着这 11G) |
| ③ 想让 merge 的 **L2 真跑** | +62G | **≈ 358G** | `minimax_h3_fl2va_bf16.safetensors` **66 280 487 368 B**(步骤 8.6) |
| ④ 想复现已判死的 SGLang TE | +34G | **≈ 392G** | `models_te_fp8/Qwen3-VL-32B-Instruct-FP8` |

**建议直接按 `df -h /` ≥ 420G 可用来准备**(留出 merge 过程和临时文件的余量)。
只做 ①+② 的话 ≥ 320G 也够。
本机采集时 `/` 是 1.9T 总量 / 已用 475G / 可用 1.3T,余量充裕。

> ⚠️ **项目根路径不可更改**:`scripts/h3_switch_5090.sh:20`、`scripts/launch_comfy.sh:8`、
> `launch_comfy_5090_turbo.sh:5`、`launch_comfy_5090_tlora.sh:6` 全是 `BASE="$HOME/data/dropbox/CV/h3"` 写死;
> `scripts/h3_generate.py:190` 是 `REMOTE_BASE = "data/dropbox/CV/h3"`(相对远端 `$HOME`)。
> **换目录整条产线就断**,而且断得很安静(switch 脚本找不到文件直接退)。要换必须同时改这 5 处。

### 步骤 0 —— 前提 B:控制端(本项目里是 Mac)

步骤 10.6 / 10.8 的 `scripts/h3_generate.py` **不在 GPU 机上跑**,它在控制端跑、通过 ssh 驱动 GPU 机。
控制端需要:

```bash
# h3_generate.py:563 会逐个检查这 4 个,缺一个直接退出
for t in ffmpeg ffprobe ssh scp; do command -v $t || echo "MISSING: $t"; done

# ~/.ssh/config 里必须有这个别名 —— h3_generate.py:37 的 HOSTS["5090"]["ssh"] 写死了它
ssh -o BatchMode=yes popos-5090 'echo SSH_OK; ls -d ~/data/dropbox/CV/h3'
```

- **`ffmpeg` / `ffprobe`**:控制端要做首尾帧的画布统一(`h3_generate.py:314` 用 ffprobe 读尺寸,`:336` 用 ffmpeg
  做 `scale=…:force_original_aspect_ratio=increase,crop=…`)。**GPU 机上装没装 ffmpeg 与此无关**
  (本机 `/usr/bin/ffmpeg` 是有的,但产线用不到它)。
- **ssh 别名必须叫 `popos-5090`**:写死在 `HOSTS["5090"]["ssh"]` 里,改主机名要改代码。
- **远端根必须恰好是 `$HOME/data/dropbox/CV/h3`**:理由同上面那条 ⚠️。

**为什么钉驱动 ≥580.65**:CUDA 13.0 运行时的最低驱动要求。低于它,`torch 2.11.0+cu130` 会在
`torch.cuda.init()` 抛 `The NVIDIA driver on your system is too old`。本机 580.173.02 满足。

> ⚠️ **没有 P2P**:不要再尝试 ComfyUI-MultiGPU 跨卡方案。历史实测在 comfy-kitchen 环境下**必崩**
> (CPU 中转与 dlpack 不兼容,illegal memory access,纯 fp16 VAE 也崩)。**别再试。**

> ⚠️ **GPU0 带显示输出**(约 0.5 GiB 常驻)。若做贴近 32.6G 上限的实验,用 GPU1,或先退出桌面会话。

安装 miniconda(本机是 `/home/isaac/miniconda3`,conda 26.7.0),然后:

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
mkdir -p ~/data/dropbox/CV/h3
cd ~/data/dropbox/CV/h3
git clone https://github.com/stargazerscripter123/H3_inference.git .   # 仓库根 = 项目根
mkdir -p inputs outputs workflows logs run src
```

### 步骤 1 —— 建三个 conda 环境

```bash
conda create -y -n h3_comfy_NV_py312  python=3.12.13
conda create -y -n h3_vllm_NV_py312   python=3.12.13
conda create -y -n h3_sglang_NV_py312 python=3.12.13
```

> ⚠️ **钉到 `3.12.13` 而不是 `3.12`**:`python=3.12` 拿到的是当时 conda 通道里最新的 3.12.x,
> 半年后建的环境就和本档案记录的不是同一个 patch 版本了。本机三个环境实测**全是 3.12.13**。
> 本文档别处都坚持精确钉版,这里也一样。

**为什么三个环境而不是一个**:三条产线的 `transformers` / `diffusers` / `numpy` / `pydantic` 版本互不兼容
(实测 comfy 用 transformers 5.14.1 + numpy 2.4.4;sglang 用 transformers 5.12.1 + pydantic 2.14.0b1 预发布)。
合并必然互相降级。

**为什么是 python 3.12**:三个环境实测都是 3.12.13;vllm 0.26.0 的 wheel 是 `cp38-abi3`(兼容),
但 ComfyUI pin 的那一版依赖树在 3.12 上验证过 —— 换 3.13 属未验证区。

### 步骤 2 —— ComfyUI 环境(顺序不能颠倒)

```bash
ENVPY=$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python

# 2.1 先装 torch,再装 ComfyUI requirements(三个都钉版:实测 2.11.0+cu130 / 0.26.0+cu130 / 2.11.0+cu130)
"$ENVPY" -m pip install --no-input torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 2.2 ComfyUI 钉死 commit
PIN=57500fc5bc92566a63f2046824f522cd55c335ca
mkdir -p ~/data/dropbox/CV/h3/ComfyUI && cd $_
git init -q
git remote add origin https://github.com/comfyanonymous/ComfyUI.git
git fetch --depth 1 origin "$PIN"
git checkout -q FETCH_HEAD

# 2.3 requirements(里面自带 comfy-kitchen==0.2.26 / comfy-aimdo==0.4.11)
"$ENVPY" -m pip install --no-input -r requirements.txt
"$ENVPY" -m pip install --no-input "huggingface_hub[cli]" hf_transfer modelscope
```

**为什么先 torch 后 requirements**:`ComfyUI/requirements.txt` **第 4 / 6 / 7 行**的 torch / torchvision / torchaudio
**没有版本约束**(第 5 行是 `torchsde`,不是 torch 三兄弟之一),先跑它会从默认 PyPI 拉一个未必带 `+cu130` 的变体。
先钉死再让 requirements 看到「已满足」。

**为什么钉 `57500fc5`**:这是 ComfyUI 首次支持 MiniMax-H3 的 merge commit(`feat: Support MiniMax-H3 (CORE-375) #15224`,v0.29.0)。
`doc/high_level.md` 明确写了不能无版本跟踪 master。

**为什么 comfy-kitchen 0.2.26 / comfy-aimdo 0.4.11 必装**:它们提供 int8-convrot / nvfp4 的量化算子
(实测启动日志里 `comfy_kitchen backend cuda` 报告 `dequantize_int8_convrot_weight` / `scaled_mm_nvfp4` 等能力)。缺了整条 int8 产线不存在。

### 步骤 3 —— 三个 ComfyUI custom node(全部钉 commit)

```bash
cd ~/data/dropbox/CV/h3/ComfyUI/custom_nodes

git clone https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache.git
git -C ComfyUI-MiniMaxH3-TeaCache checkout 4cbb50d69c73a19a5d6ec42c5aec1989d5a04b6f

git clone https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git
git -C ComfyUI-MiniMax-H3-Turbo checkout 55fee864dd7b2976b1c4ce3c3d5f7968f181409f

git clone https://github.com/pollockjj/ComfyUI-MultiGPU.git
git -C ComfyUI-MultiGPU checkout b51c99a525e9607e43545ee2a8b7694c74a4775a
```

- **TeaCache `4cbb50d`**:有损加速节点,版本变动会改变有损程度 → benchmark 不可比。
- **Turbo `55fee864`**:**唯一**能正确加载 larryvrh 裸键 LoRA 并在运行时注入 51 个 adaln 的实现。零 pip 依赖。
  clone 时会一并带下 `h3_silu_temb_grid.safetensors`(5.25 MiB),不需要单独下载。
- **MultiGPU `b51c99a5`**:跨卡路线虽已放弃,但当前所有 benchmark 都是在它 patch 过 `model_management` 的前提下测的
  → **为了可比性照装**。如果决定不装,必须把 5090 的全部 ComfyUI 数字重测。

> ⚠️ **绝对不能用 ComfyUI 核心的 `LoraLoaderModelOnly` 加载 Turbo LoRA**:
> `comfy/lora.py` 的 `model_lora_keys_unet` 只认 `diffusion_model.<k>` / `lora_unet_<k>`,没有 MiniMaxH3 分支;
> larryvrh 的 LoRA 是裸键 `blocks.0.attn.qkv_proj.lora_A.weight` → **0/518 命中且不报错**,出片等于「6 步的 base」。
> 必须用作者节点 `MiniMaxH3TurboLoRA`。验证办法见步骤 9。

### 步骤 4 —— vLLM 环境(生产 serving)

```bash
VLM_PY=$HOME/miniconda3/envs/h3_vllm_NV_py312/bin/python

# 4.1 base wheel(它会带来 torch 2.11.0+cu130)
"$VLM_PY" -m pip install --no-input -q uv
"$VLM_PY" -m uv pip install --python "$VLM_PY" "vllm==0.26.0" --torch-backend=auto

# 4.2 拿 PR#5910 的源码,钉死到 b18eeff2
cd ~/data/dropbox/CV/h3/src
git clone https://github.com/vllm-project/vllm-omni.git vllm-omni-pr5910
cd vllm-omni-pr5910
git fetch origin pull/5910/head:pr5910-head
git checkout -b pr5910-stride-fix b18eeff22a30b426060f44f734d8b16341b02959
```

**为什么钉 `b18eeff2` 而不是「PR#5910 的 head」**:实测该 PR 被 force-push 过 ——
本机同时留着 `9f921476`(旧计划稿,**不是** `b18eeff2` 的祖先)、`b18eeff2`(生产基线)、
`0bcc1a9f`(其后的 docs 提交),runpods 上又是另一个 `1a9b9c2c`。**只有 commit 是可复现的坐标,PR 号不是。**

**为什么用 vllm-omni 源码而不是 wheel**:0.26.0 的 wheel 里 H3 缺 `frame_indices` 首尾帧支持(8-05 才 merge)。

### 步骤 5 —— 打 stride 补丁并固化成本地提交

```bash
cd ~/data/dropbox/CV/h3/src/vllm-omni-pr5910
git apply ~/data/dropbox/CV/h3/scripts/pr5910_resident_stride_fix.patch
git commit -am "fix(dlo): keep FP8 transposed-view stride when repointing resident layers"
# 本机固化结果 = 070096bd(内容等价即可,哈希不必相同)

# 判定(必须用类内限定版!)
sed -n '/class PinnedResidentLayerGroup/,/def offload/p' \
    vllm_omni/diffusion/offloader/distributed_layerwise_backend.py | grep -q as_strided \
    && echo APPLIED || echo MISSING
```

**为什么必须打**:`_shard_and_pin` 按物理(列主序)顺序打包在线-FP8 的转置权重并保存 stride;
streamed 路径 `prefetch_layer` 用 `as_strided` 还原是对的,常驻组的 `load()` 却用 `.view()` 按行主序重解释,
导致常驻 block 的所有 linear 权重**值级乱序**。宽松 kernel 静默出纯噪声,严格 kernel 直接崩。
round-1 曾据此误判为「sm_120 FP8 全线不可用」。

> ⚠️ **判定绝对不能用 `git diff`**:补丁一旦 commit,worktree 就是干净的,`git diff --quiet` 会反过来报「没打」。
> ⚠️ **判定也不能用整文件 `grep as_strided`**:实测未打补丁的 `b18eeff2` 里,那个文件**本来就有 2 处 `as_strided`**
> (行 299 / 412,streamed 路径)。整文件 grep **恒真**。
> `scripts/h3_switch_5090.sh`、`scripts/h3_switch_runpods.sh`、`scripts/setup_runpods.sh` 目前用的都是**错的整文件版**
> (`scripts/post_setup.sh` / `post_setup2.sh` 用的是对的类内版)。**照抄前先看清楚。**
> ⚠️ 源码被 reset / 升级后**必须重打**;`scripts/pr5910_resident_stride_fix.patch` 是权威副本。

### 步骤 6 —— editable 安装 vllm-omni

```bash
cd ~/data/dropbox/CV/h3/src/vllm-omni-pr5910
SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0 \
  "$VLM_PY" -m pip install -e . --no-build-isolation
"$VLM_PY" -m pip install --no-input aenum
```

**为什么 `SETUPTOOLS_SCM_PRETEND_VERSION`**:构建靠 setuptools_scm 从 git 取版本。
如果源码是 rsync 过来的(丢了 `.git`),版本会回退成 `0.1.dev1+…` 而与 `vllm 0.26.0` 报 major/minor 不匹配警告并可能构建失败。
本机 `src/vllm-omni` 与 `src/sglang` 就是活体标本。

**为什么 `--no-build-isolation`**:构建脚本会 import 自身,需要依赖(vllm)在场。

**为什么要单独装 `aenum`**:`vllm_omni/patch.py:7` 直接 `from aenum import extend_enum`。
`logs/fix_installs.log` 里有现场:`ModuleNotFoundError: No module named 'aenum'`,整个 `import vllm_omni` 挂掉。

> ⚠️ **不要用 `rm -rf site-packages/<pkg>` 卸包**:会残留断损元数据,让 `--omni requires vllm-omni` 假报。用规范 `pip uninstall`。
> ⚠️ **不要把多个 pip 安装合并成一行**:历史上一行合并在 nvcc 的 sdist 构建上失败,把 uv/ninja 一起带崩,
> 后续所有步骤静默跳过(`scripts/fix_5090_installs.sh` 整份就是那次事故的修复)。nvcc 系一律
> `pip install --only-binary :all: nvidia-cuda-nvcc-cu13 nvidia-cuda-runtime-cu13`。

**(可选)第二个补丁 —— `_ASYNC_OUTPUT_TIMEOUT`**:

```bash
grep -n "_ASYNC_OUTPUT_TIMEOUT" vllm_omni/diffusion/diffusion_engine.py   # :58 定义, :326/:793 使用
# 本机实测仍是 30.0(未改)。判据见第 8 节 —— 只有在遇到「HTTP 500 + 空 message」时才需要改大。
```

### 步骤 7 —— (可选)SGLang 环境

**这条路线已判死**(PR33681 TP2 在线 FP8 需先整载 33G BF16 shard > 31.3G,加载期 OOM;
TP1 对照也 OOM,TE 构造期需整载 51.5G BF16)。**只有想重新验证这个结论时才装。**

```bash
SGL_PY=$HOME/miniconda3/envs/h3_sglang_NV_py312/bin/python
"$SGL_PY" -m pip install --no-input -q uv ninja
"$SGL_PY" -m pip install --only-binary :all: nvidia-cuda-nvcc-cu13 nvidia-cuda-runtime-cu13

cd ~/data/dropbox/CV/h3/src
git clone https://github.com/sgl-project/sglang.git sglang-pr33681
cd sglang-pr33681
git checkout -b h3-fp8-te 60b9e5149930acd048936fcb212777410932f227

SGLANG_BUILD_RUST_EXTS=none \
  "$SGL_PY" -m uv pip install --python "$SGL_PY" -e "python[diffusion]" --prerelease=allow

# JIT 需要的无版本 symlink
LIB=$HOME/miniconda3/envs/h3_sglang_NV_py312/lib/python3.12/site-packages/nvidia/cu13/lib
ln -sfn "$LIB/libcudart.so.13" "$LIB/libcudart.so"
```

**为什么 `SGLANG_BUILD_RUST_EXTS=none`**:不构建 Rust 扩展(diffusion 路径不需要,构建又慢又容易失败)。
**为什么要那个 symlink**:系统没有 CUDA toolkit,JIT 链接 `-lcudart` 只能靠 pip wheel 里的库,而它只提供带版本号的 `.so.13`。

### 步骤 8 —— 下权重

#### 8.0 先做两件事,否则后面每一步都会以难查的方式失败

**(1) 写 HF token。** 这一步必须在 8.1 之前做完:

```bash
mkdir -p ~/.cache/huggingface
printf %s '<你的 token>' > ~/.cache/huggingface/token     # 取自 credentials/HF.md,不入库
chmod 600 ~/.cache/huggingface/token
ls -l ~/.cache/huggingface/token                          # 本机实测 37 字节
# 等价做法: "$HF" auth login   (HF 变量见下面第 (2) 条)
```

- **三个 repo 都不是 gated**(实测 HF API `gated=False`:`MiniMaxAI/MiniMax-H3`、`Comfy-Org/MiniMax-H3`、
  `larryvrh/MiniMax-H3-Turbo-Lora`;匿名 `curl -I` 取 fp8_scaled / bf16 都返回 200)。
  **token 不是为了过权限,是为了不被限速** —— 匿名下载历史上 144G 卡到日流量 ~300G 上限。
- ⚠️ **`scripts/dl_fp8_auth.sh` 在没有 token 时不会报错退出**:它第 6 行是 `TOKEN=$(cat ~/.cache/huggingface/token)`,
  `set -u` 管不到命令替换失败,于是带着**空 Bearer 头**空转 —— `for i in $(seq 1 40)` × `--retry 3`,
  要转很久才打印 `FP8_DOWNLOAD_FAILED`。**极难排查,所以 token 一定要先写。**

**(2) 把 `hf` 的绝对路径拿出来。**

```bash
HF=$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/hf
"$HF" version    # 确认能跑
```

> ⚠️ **不能直接敲 `hf`**:`hf` CLI 是步骤 2.3 装进 `h3_comfy_NV_py312` 这个 conda env 的,
> 而本节全程**不 `conda activate`**(其余步骤也都用 `$ENVPY` / `$VLM_PY` / `$SGL_PY` 绝对路径)。
> 实测在未激活的 shell 里 `which hf` 与 `which huggingface-cli` **都是 NOT_ON_PATH**,
> 裸敲会 `hf: command not found`,145G 基座和 LoRA 都下不下来。
> 想省事也可以 `conda activate h3_comfy_NV_py312`,但**后续步骤仍然要用绝对路径**(它们跨 env)。

#### 8.1–8.6 正式下载

```bash
cd ~/data/dropbox/CV/h3
HF=$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/hf     # 见 8.0(2)

# 8.1 ComfyUI 单文件组(int8 DiT + nvfp4 TE + int8 TE + 两个 VAE),按字节数校验+断点续传
bash scripts/download_h3.sh 5090

# 8.2 (可选,A/B 用)pruned FP8_scaled DiT —— 走带 token 的 curl(限速规避,非鉴权)
bash scripts/dl_fp8_auth.sh

# 8.3 官方 BF16 基座(145G)。两条路,任选:
#   (a) 局域网内已有别的机器有 → rsync(1Gbps 约 25min)
#   (b) 从 HF 下 —— 注意必须连根级配置一起下,不能只要 FL2VA:
#       (--include 必须【重复写】,不能空格并列 —— 见下方 ⚠️)
HF_XET_HIGH_PERFORMANCE=1 "$HF" download MiniMaxAI/MiniMax-H3 \
    --include "FL2VA/**" --include "*.json" --include "README.md" \
    --local-dir models_official/MiniMax-H3 \
    --revision b3c7290e66afdf293bef3b9077b7a266ef421f34
# 校验: models_official/MiniMax-H3/FL2VA/transformer 下必须是 13 个 safetensors 分片
ls models_official/MiniMax-H3/FL2VA/transformer/*.safetensors | wc -l   # 期望 13
ls models_official/MiniMax-H3/model_index.json                          # 必须存在(SGLang 靠它解析仓库)

# 8.4 (仅 SGLang 路线需要) HF cache 桥接,让 SGLang 免重复下载 145G
#     vLLM 与 ComfyUI 都不读这个 cache,不跑 SGLang 可整步跳过。
H=~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3
SHA=b3c7290e66afdf293bef3b9077b7a266ef421f34
SRC=~/data/dropbox/CV/h3/models_official/MiniMax-H3
rm -rf "$SRC/.cache"                            # ← 先清掉 hf download 的残留,否则会被一起桥进 cache
mkdir -p "$H/refs" "$H/snapshots/$SHA"          # ← refs/ 必须显式建,否则下面写 refs/main 会 No such file
cp -rs "$SRC"/. "$H/snapshots/$SHA"/            # 目录建成真实目录、目录内逐文件 symlink
echo -n "$SHA" > "$H/refs/main"
# 校验(应与 5.1 #11 的实测结构一致):
ls -la "$H/snapshots/$SHA" | head             # 顶层: 12 个真实目录 + 3 个根级 json/README symlink
ls -la "$H/snapshots/$SHA/FL2VA/transformer" | head   # 内部: 逐文件 symlink 指回 models_official

# 8.5 Turbo LoRA(只下那一个文件!整仓 22 个文件 = 白下 22G)
"$HF" download larryvrh/MiniMax-H3-Turbo-Lora \
    --include "minimax_h3_turbo_v4_step600_ema.safetensors" \
    --revision afc0346 --local-dir loras/
sha256sum loras/minimax_h3_turbo_v4_step600_ema.safetensors
#   期望 5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3
ln -sfn ~/data/dropbox/CV/h3/loras/minimax_h3_turbo_v4_step600_ema.safetensors \
        ComfyUI/models/loras/minimax_h3_turbo_v4_step600_ema.safetensors

# 8.6 (强烈建议) BF16 单文件 —— 步骤 9 的 L2 校验 oracle,62 GiB
#     不下这个,步骤 9 的 L2 会被静默跳过(见步骤 9 的 ⚠️)。
curl -L -C - --retry 5 --retry-delay 5 \
  -o ComfyUI/models/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors
stat -c %s ComfyUI/models/diffusion_models/minimax_h3_fl2va_bf16.safetensors
#   必须精确等于 66280487368 —— 不等就是没下完,重跑上面那条 curl(-C - 会续传)
```

> ⚠️ **8.6 绝对不要用 `scripts/download_bf16.sh`**。那是 **6000a 专用**脚本:
> `STORE="/home/isaac/Data/h3_weights"` 写死(6000a 的第二块 NVMe),而 5090 上
> `/home/isaac/Data` 是**另一块盘、另一份 Dropbox**(见第 1 节);它还会
> `while pgrep -f download_h3.sh; do sleep 60; done` 死等一个不存在的进程,
> 并额外拉一份 5090 用不到的 51.5G TE BF16。**用上面那条 `curl` 就够了。**

> ⚠️ **8.3(b) 的 `--include` 不能只写 `"FL2VA/**"`**:那样顶层就只有 `FL2VA/` 一项,
> `model_index.json` / `modular_model_index.json` / `README.md` 全部缺席,
> 而 SGLang 的 `--model-path MiniMaxAI/MiniMax-H3` **正是靠仓库根的 `model_index.json` 解析的**,8.4 的桥接也就没了意义。
> 本机 `models_official/MiniMax-H3/` 实测**下的是整仓**(根级 `model_index.json` 2936 B、`Ref2VA/` 29M、
> `scheduler/` `vae/` `transformer/` 等一应俱全),只是大文件集中在 `FL2VA/`。
> 想完全照抄本机,把 `--include` 整个去掉即可(代价是多下 `Ref2VA` 等约 60M)。

> ⚠️ **多个 `--include` 必须重复写这个选项,不能空格并列** —— 这是本轮实测踩到的一个静默陷阱。
> `hf` 1.26.0 的用法是 `hf download [OPTIONS] REPO_ID [FILENAMES]...`,`--include` 是**可重复的单值选项**。
> 写成 `--include "FL2VA/**" "*.json" "README.md"`,后两个会被当成**位置参数 FILENAMES**,于是:
>
> ```
> warnings.warn("Ignoring `--include` since filenames have been explicitly set.")
> ```
>
> —— **`--include` 整个被忽略,FL2VA(135G)一个文件都不会下**,只会去下两个叫 `*.json` / `README.md` 的字面文件。
> 而这只是一条 warning,不是错误。正确形式:`--include A --include B --include C`。

> ⚠️ **8.4 用 `cp -rs` 而不是顶层 `for f in …; do ln -sfn` 循环**:后者产生的是「顶层全是 symlink(含 `FL2VA` 目录 symlink)」,
> 与本机实测结构(5.1 #11:顶层 12 个**真实目录** + 3 个根级 symlink,目录内部再逐文件 symlink)**不一致**。
> `cp -rs` 正好复刻真实结构。另外 `mkdir -p "$S"` **只建 `snapshots/<sha>`,不建 `refs/`** ——
> 新机上 8.3 用的是 `--local-dir`,根本不会生成 hub cache 条目,所以不显式 `mkdir "$H/refs"`
> 那行 `echo -n … > refs/main` 必然 `No such file or directory`。

> ⚠️ **`HF_HUB_ENABLE_HF_TRANSFER` 已被弃用**(本机 `logs/download_modelopt.log` 里就有这条 FutureWarning),
> 走退化路径只有 ~28MB/s。用 **`HF_XET_HIGH_PERFORMANCE=1`**。`scripts/setup_runpods.sh` 里还是旧的,是已知待修缺陷。
> ⚠️ **audio VAE 必须 fp32**(BF16 会让音量 −20dB),不要图省事换 fp16。
> ⚠️ **`hf download --local-dir` 会在目标目录下留一个 `.cache/huggingface/`**(本机就剩了 **11G** xet 半成品,见 5.4)。
> 下完确认无误后可 `rm -rf models_official/MiniMax-H3/.cache` 回收。

**TE FP8(仅 SGLang 路线需要)**:HF 上下载会限速卡死(历史上卡在 13G),换 **ModelScope**
(本机 `logs/download_te_ms.log`:19 个文件、2 小时 13 分、单文件 ~1.2–2.0 MB/s)。
vLLM 路线做在线量化,**不需要这份权重**。

### 步骤 9 —— 合并 Turbo LoRA

```bash
$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python scripts/merge_turbo_lora.py \
  --lora   loras/minimax_h3_turbo_v4_step600_ema.safetensors \
  --lora-revision afc0346 \
  --base   models_official/MiniMax-H3/FL2VA \
  --dst    models_merged/MiniMax-H3-Turbo-v4s600ema \
  --strength 1.0 \
  --comfy-single ComfyUI/models/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  2>&1 | tee logs/merge_turbo.log
```

**为什么要 merge 而不是运行时挂 LoRA**:vLLM-Omni 的 runtime LoRA 对 fused 层(qkv 21504 行 / fc1 28672 行)
会 warning 后**跳过**(`diffusion/lora/manager.py:697-706`),蒸馏静默失效。vLLM 只走 merged checkpoint 路线。

**耗时参考(本机实测)**:merge ~90s(13 shard × ~7s),L1 校验 ~56s,总计约 2.5 分钟。

⚠️ 上面这条命令**假定你已经做过步骤 8.6**(BF16 单文件在位)。**如果没做,先读下面这段再跑。**

**期望输出(做过 8.6,L2 真跑)**:
```
all 259/259 LoRA targets applied
verify L1 passed: 259 modified + 276 unmodified = 535 tensors bit-checked
verify L2 passed: Comfy single-file == reorder(HF disk) bit-exact on 6 qkv + 5 plain layers
verify L3 passed: worst min_row_cos=0.999999, worst relL2=9.65e-04
DONE -> …/models_merged/MiniMax-H3-Turbo-v4s600ema (manifest + .complete written)
```

**如果决定不下那 62G**,就把命令里的 `--comfy-single` 一行改成**显式**的:

```bash
  --comfy-single /nonexistent \      # 明写:本机就是这么跑的,L2 主动放弃
```

**期望输出(L2 跳过)** —— 与上面的差别只有一行,而且是**警告不是错误,退出码仍是 0**:
```
verify L1 passed: 259 modified + 276 unmodified = 535 tensors bit-checked
verify L2 SKIPPED (no --comfy-single) — layout not independently confirmed!
verify L3 passed: worst min_row_cos=0.999999, worst relL2=9.65e-04
```

> ⚠️ **这是本文档最容易被照抄坏的一处,机制说清楚**:
> `scripts/merge_turbo_lora.py:264` 是 `if args.comfy_single and os.path.exists(args.comfy_single):` ——
> **`--comfy-single` 指向的路径不存在时,L2 静默跳过**,只在日志第 280 行留一句 `verify L2 SKIPPED`,
> 退出码 0,manifest 照写「L2 Comfy single-file layout oracle」。
> **本机的 `models_merged/` 就是这么产出的:L2 一次都没跑过**
> (`logs/merge_turbo.log` → `[08:48:20] verify L2 SKIPPED (no --comfy-single) — layout not independently confirmed!`)。
> 所以判据是:**日志里必须有 `verify L2 passed` 这一行**;只要看到 `verify L2 SKIPPED`,
> 就说明布局没被独立确认过 —— 那不是「输出正常」。
>
> ⚠️ **manifest 的 `verification` 字段不能用来判断 L2**,它是静态模板。判据只有 `logs/merge_turbo.log`。
>
> ⚠️ **manifest 的 `merge_script_sha256` 会与 5.3 表里记的 `b034c41b…` 不同**,它写的是**你机器上这份脚本**的哈希
> (仓库当前版本是 `70cdfe9d7b1e…`)。**这是预期内的**,原因见 5.3 那条注。
>
> **没做 8.6 时,替代证据链是什么**:与另一台机器(本项目里是 6000a)比对
> `base_transformer_index_sha256` / `lora_sha256` / `259` 与 `535` 计数 —— 三项一致即认为位级等价。
> 注意 `merge_script_sha256` **已经不能作为这条链的一环**(见 5.3)。
> 这条链只证明「输入相同」,不证明「磁盘布局解释正确」。**手上没有可比对的机器时,请务必做 8.6 让 L2 真跑。**

### 步骤 10 —— 冒烟验证

```bash
# 10.1 三个环境的 torch 都能看到两张 sm_120 卡
for e in h3_comfy_NV_py312 h3_vllm_NV_py312 h3_sglang_NV_py312; do
  $HOME/miniconda3/envs/$e/bin/python -c \
    "import torch;print('$e',torch.__version__,torch.version.cuda,torch.cuda.device_count(),torch.cuda.get_arch_list()[-1])"
done
# 期望(三行完全一样 —— torch.__version__ 看不出通道差异):
#   h3_comfy_NV_py312  2.11.0+cu130 13.0 2 sm_120
#   h3_vllm_NV_py312   2.11.0+cu130 13.0 2 sm_120
#   h3_sglang_NV_py312 2.11.0+cu130 13.0 2 sm_120

# 10.1b 通道差异只在 pip 元数据里(别拿 torch.__version__ 找它,永远找不到)
for e in h3_comfy_NV_py312 h3_vllm_NV_py312 h3_sglang_NV_py312; do
  printf '%s ' "$e"; $HOME/miniconda3/envs/$e/bin/python -m pip show torch | grep '^Version'
done
# 期望:
#   h3_comfy_NV_py312  Version: 2.11.0+cu130     ← download.pytorch.org/whl/cu130
#   h3_vllm_NV_py312   Version: 2.11.0+cu130     ← 同上
#   h3_sglang_NV_py312 Version: 2.11.0           ← PyPI 默认通道,正常,功能等价

# 10.2 vllm-omni 解析到正确的源码树 + H3 模型在场
$HOME/miniconda3/envs/h3_vllm_NV_py312/bin/python -c \
 "import vllm,vllm_omni,pathlib;print(vllm.__version__);print(vllm_omni.__file__);\
  print('minimax_h3:',(pathlib.Path(vllm_omni.__file__).parent/'diffusion/models/minimax_h3').exists())"
# 期望: 0.26.0 / …/src/vllm-omni-pr5910/vllm_omni/__init__.py / minimax_h3: True

# 10.3 stride 补丁(类内限定判据)
sed -n '/class PinnedResidentLayerGroup/,/def offload/p' \
  ~/data/dropbox/CV/h3/src/vllm-omni-pr5910/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py \
  | grep -q as_strided && echo STRIDE_APPLIED || echo STRIDE_MISSING
# 期望: STRIDE_APPLIED

# 10.4 LoRA 体检(Gate G0)—— 注意是【位置参数】,不是 --lora
$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python scripts/check_turbo_lora.py \
  loras/minimax_h3_turbo_v4_step600_ema.safetensors
# 期望: 259 对键 / 518 tensors / 全 BF16 / shape 全部匹配,exit 0
#
# ⚠️ 两个脚本的参数形式【不一样】,别混:
#     check_turbo_lora.py  <lora>          位置参数 (p.add_argument("lora"), 第 44 行)
#     merge_turbo_lora.py  --lora <lora>   选项
#   写成 check_turbo_lora.py --lora <path> 会被 argparse 直接拒绝:
#     usage: check_turbo_lora.py [-h] lora
#     error: unrecognized arguments: --lora

# 10.5 merged checkpoint 完整性
ls -l models_merged/MiniMax-H3-Turbo-v4s600ema/.complete          # 必须存在,否则 switch 拒绝启动
ls models_merged/MiniMax-H3-Turbo-v4s600ema/FL2VA/transformer/*.safetensors | wc -l   # 期望 13

# 10.6 ComfyUI 端到端 —— 【在控制端(Mac)上跑,不是在 GPU 机上】,前提见步骤 0 前提 B
scripts/h3_generate.py --host 5090 --profile comfy-int8-original-1c --first a.png --last b.png --prompt "..."
# 期望: 124 帧 / 5.167s / 24fps / AAC 立体声 32kHz
# ⚠️ 这条是 30 步【无加速基线】,warm 约 95–110s(doc/speedup_5090_results.md 的「参照」行)。
#    别拿它对 25s —— 那是 comfy-int8-turbo-1c 的数(NFE4 warm 25.0s / 冷启 45.0s,
#    见 doc/turbo_lora_results.md 第九节)。想快速冒烟就用 --profile comfy-int8-turbo-1c --nfe 4。

# 10.7 Turbo LoRA 真的生效(否则等于「6 步的 base」,画面看不出来)
grep -n "adaln injected at run time" logs/comfyui_tlora.log
# 期望看到: 208 backbone modules, 158 bypass adapters, … 50 int8 fc2 via merge + 51 adaln injected at run time
grep -n "forward_owner=BypassForwardHook" logs/comfyui_tlora.log | head -1
# 期望: BypassForwardHook => lora ACTIVE

# 10.8 vLLM TP2 端到端
bash scripts/h3_switch_5090.sh vllm            # ← 在 GPU 机上跑;冷启约 1 分钟
head -3 logs/vllm_server.log                    # ← 在 GPU 机上看。⚠️ 头里的 stride_patch=APPLIED 是恒真的假证据(见 4.2 冲突 2)
scripts/h3_generate.py --host 5090 --profile vllm-fp8-original-tp2 ...   # ← 在控制端(Mac)上跑
# 期望: warm 46.9 / 46.8 / 46.6 s,峰值显存约 23.95 G/卡
```

> ⚠️ **`/health` 早于权重加载就绪** —— 切换后的第一单必然是 warmup,**不计时**。
> ⚠️ **benchmark 必须声明冷热态**:5090 ComfyUI(`comfy-int8-turbo-1c` @ NFE4)冷 45.0s / 热 25.0s(差近一倍)。用 `scripts/h3_eval.py` 的「1 warmup 不计 + N timed」协议。
> ⚠️ **ComfyUI 会缓存整图**:同参数重跑必须换 seed 或 prompt 后缀;TeaCache 另需对 `rel_l1_thresh` 做 1e-9 级 per-run 抖动(runner 已内置,改 runner 时保留)。

### 步骤 11 —— 生产配置速查(vLLM TP2)

```text
PR5910 @ b18eeff2 + stride patch(本地分支 pr5910-stride-fix)
TP2 + TE-TP2 | global online FP8(per-tensor) | DLO no-allgather resident=50(H3 恰 50 层 = 全常驻)
CUDNN_ATTN(勿用 TORCH_SDPA,慢约 18s → 64.8s)| enforce-eager
VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel(sm_120 上禁 Cutlass)
warm 46.9 / 46.8 / 46.6 s,峰值 23.95 G/卡
```

---

## 8. 已知坑与注意事项

### 8.1 本次采集新发现的(档案里没有的)

1. **`h3_switch_5090.sh` 的 stride 探测恒真**(第 4.2 节)。日志头 `stride_patch=APPLIED_as_strided`
   **不能作为补丁存在的证据**。真判据:类内限定的 `sed … | grep -q as_strided`。
2. **`_ASYNC_OUTPUT_TIMEOUT = 30.0` 在 5090 上根本没打**,与档案「三机必打」的说法冲突。
   判据:只有当客户端收到 **HTTP 500 且 body 是 `{"error":{"message":"Video generation failed:"}}`(空 message)** 时才需要改大;
   5090 的生产组合(~47s)历史上一直能过,因为 30 秒卡的是**两次 yield 之间**的间隔。
   **不要一看到 47s > 30s 就断定要改。**
3. **GPU0 是显示卡**,常驻约 0.5 GiB,而 :8189/:8190 两个 worker 恰恰在 GPU0。
4. **`src/vllm-omni` 与 `src/sglang` 不是 git 仓库**(rsync 丢了 `.git`),上游 commit 只能从
   `logs/fix_installs.log`(`a874b8e09`)与 `logs/install_pr33681.log`(`407a65d3c`)反推。
5. **SGLang env 当前 editable 指向的是「已判死」的 PR33681 分支**,main 那份(`src/sglang`)没被安装。
6. **`models_modelopt/MiniMax-H3-FP8` 是残缺下载**(38/74 文件,5.0G),无人引用。
7. **merge manifest 的 `verification` 字段是静态模板**,本机这份 L2 实际被跳过。真相只在 `logs/merge_turbo.log`。
8. **ComfyUI-MultiGPU 仍在每个 worker 里 monkey-patch 核心**,尽管跨卡路线已放弃。
9. **`launch_comfy.sh` 没设 `CUDA_DEVICE_ORDER`**,另两个 launcher 设了 `PCI_BUS_ID` —— 不一致。
10. **机器不是 Pop!\_OS 而是 Ubuntu 24.04.4**;内核 `7.0.0-28-generic`。
11. **`merge_manifest.json` 的 `merge_script_sha256`(`b034c41b…`)与仓库现版本脚本(`70cdfe9d7b1e…`)不一致** ——
    产出那份 checkpoint 的脚本已不在树上。**它不能再充当「L2 被跳过后」证据链的一环**(见 5.3);
    重跑步骤 9 时 manifest 会写新哈希,属正常。
12. **`models_official/MiniMax-H3/.cache/` 有 11G 未清理的 xet 半成品**,直接推翻了「基座纯粹来自 rsync」的旧推测,
    也是 145G 这个数字对不上账的原因(见 5.4)。可直接删。
13. **`/home/isaac/Data` 在 5090 上是存在的**(3.6T LUKS 盘,另一份 Dropbox),
    所以 6000a 专用的 `scripts/download_bf16.sh` 在这台机器上**不会因路径不存在而失败**,
    而是会把 113G 写进别人的盘 —— 比直接报错更危险(见第 1 节)。
14. **`torch.__version__` 三个环境全是 `2.11.0+cu130`**;通道差异只在 `pip show` / lock 文件里
    (sglang env 的 pip 元数据是裸 `2.11.0`)。旧版本档案把这条差异挂在了 `torch.__version__` 上,是错的(见 3.3)。

### 8.2 沿用自历史档案、本次已交叉验证成立的

| 坑 | 处置 | 本次验证 |
|---|---|---|
| 双 5090 无 P2P,ComfyUI-MultiGPU 跨卡必崩 | 放弃跨卡,单卡稳定配置,**别再试** | `topo -p2p r/w` 全 CNS ✅ |
| stride 补丁不打 → 静默纯噪声或崩溃 | 打补丁 + 固化到本地分支 | 生产分支已打 ✅ |
| FP8_scaled 在 5090 上比 INT8 慢 20%(3.96 vs 3.29 s/it) | 生产维持 pruned INT8 convrot | 两份权重都在,switch 默认走 int8 ✅ |
| SGLang PR33681 TP2 加载期 OOM;TP1 也 OOM | 路线判死 | env 仍指向该分支 ⚠️ |
| host RAM 125G 装不下三个 ComfyUI worker(每个 ~45G) | `comfy-tlora` 独占启动(:8190 与 :8188/:8189 互斥) | `free -g` = 125 ✅;switch 脚本已实现互斥 ✅ |
| `vllm` ↔ `vllm-turbo` 共用 :8091 会静默服错 checkpoint | `run/vllm.variant` + `run/vllm.model` 变体追踪,互切必重启 | 脚本实现完整 ✅;日志头确实写了 model/label/branch ✅ |
| flock + 守护进程 fd 泄漏死锁 | 所有守护进程启动处加 `9>&-` | `h3_switch_5090.sh` 里 **5 处** `9>&-` 全在 ✅(2 处 setsid 守护:`_launch_vllm`:94、`_launch_sglang`:157;3 处 `launch_comfy*` 调用:184/185/192) |
| ComfyUI 核心 LoraLoader 静默 0/518 命中 | 必须用 `MiniMaxH3TurboLoRA` 作者节点 | 日志证据 51 adaln 注入 ✅ |
| TeaCache 同参数第二单静默退化 | runner 对 `rel_l1_thresh` 做 1e-9 抖动 | 已在 runner 内 |
| `H3_SWITCH_LIB=1` 会被 export 继承 | 测完换 shell | — |
| ssh 双引号里的 `~` 被本地 shell 展开 | 远端路径用单引号或 `\$HOME` | **本次采集实际踩到一次**(`git -C "$HOME/..."` 报 `cannot change to '$HOME/...'`),改用 `ssh host 'bash -s' <<'EOF'` 后正常 |
| `pgrep`/`pkill` 模式要用方括号 | `'[m]ain.py'` / `'[v]llm serve'` | 本次采集全程使用 ✅ |

### 8.3 输出契约(跨机通用,复述以免漏)

- 帧数必须 `n % 17 == 5`(124 帧 = 5.167s @24fps);画布必须是 32 的倍数。
- 输出:124 帧 / 5.167s / 24fps / AAC 立体声 32kHz。
- 首帧 stretch / 尾帧 cover-crop 不对称 → 必须预处理统一画布(Mac CLI 已内置 `scale=…:force_original_aspect_ratio=increase,crop=…`)。
- 采样配方:video shift 12 / audio shift 3(引擎默认即 canonical);strength 1.0;alpha 缺省 = rank(scale 1.0)。
- 步数语义:**ComfyUI `steps == forwards`;引擎 `num_inference_steps = NFE + 1`**。两边都由 `--nfe` 统一表述。
  升级 vllm-omni / sglang 后必须重跑 `scripts/test_h3_schedule.py`(上游有开放 PR 要改这个语义)。

---

## 9. 本文档的采集方式

- **采集日期**:2026-08-09(机器本地时区 +10:00,采集窗口 21:31–21:40)。
- **采集方式**:从 Mac 仓库根用 `ssh popos-5090 'bash -s' <<'EOF' … EOF` 批量执行只读命令
  (`nvidia-smi` / `lscpu` / `free` / `df` / `lsblk` / `conda env list` / `pip list` / `pip show` /
  `pip freeze --all` / `conda env export` / `git -C … rev-parse|log|status|show` / `du -sh` / `ls -la` /
  `sha256sum` / `sed`+`grep` 判补丁 / `ss -ltn` / `pgrep -af '[m]ain.py'`),
  外加在 Mac 侧读 `scripts/` 与 `python3 scripts/h3_generate.py --list`。
  **全程只读:未修改远端任何文件、未装卸任何包、未启停任何服务、未跑任何生成任务。**
  唯一写盘动作是把清单落到 Mac 仓库的 `doc/machines/locks/`。
- **重新采集**:重复上述命令即可。最省事的做法是按第 3 节列的 6 个 lock 文件名重新导一遍
  (`pip freeze --all` + `conda env export`,记得改文件头的日期行并重跑一次密钥扫描),
  再把第 1/2/4/5 节的表格逐格用对应命令核一遍。
  **补丁状态一定要用类内限定的 `sed … | grep -q as_strided`,不要用 `git diff`,也不要用整文件 grep。**
- **重新采集时几个容易记错口径的地方**(都在本轮复核里踩过):
  - **包数**用 `pip freeze --all | wc -l`(108/230/239),**不要**用 `pip list | wc -l`(带 2 行表头);
    `conda list | grep -vc '^#'` 是另一个口径(135/257/266),两者不可混用。
  - **torch 通道差异**用 `pip show torch | grep ^Version`,**不是** `torch.__version__`(后者三个环境全带 `+cu130`)。
  - **`du -sh models_official`** 的 145G 里含 11G 的 `.cache/` 残留,报体积时要说清拆解。
  - **`merge_manifest.json` 的 `merge_script_sha256`** 记的是产出时的脚本,与当前树不一致是常态,
    核对时请同时 `sha256sum scripts/merge_turbo_lora.py` 并把两个值都写进文档。
- **配套清单**:
  - `doc/machines/locks/popos-5090.h3_comfy_NV_py312.pip-freeze.txt` / `.conda-env.yml`
  - `doc/machines/locks/popos-5090.h3_vllm_NV_py312.pip-freeze.txt` / `.conda-env.yml`
  - `doc/machines/locks/popos-5090.h3_sglang_NV_py312.pip-freeze.txt` / `.conda-env.yml`
