# doc/machines — 机器档案总览

**这个目录记录「每台 GPU 机器实际是什么样」——硬件、驱动、Python 环境、第三方源码与补丁、
权重落盘、运行期环境变量、从零复现步骤。**
三份机器档案都是 2026-08-09 在各机上实跑只读命令采集的,凡推测都在原文里显式标注。

**什么时候看这里**:

| 场景 | 先看 |
|---|---|
| 拿到一台新机器,不知道装哪套轮子 / 能不能跑 serving | 本文 §3 决策树 |
| 想知道三台机器哪里一样、哪里不一样 | 本文 §2 |
| 要在某台机器上复现/排障/重装 | 对应的 `popos-5090.md` / `popos-6000a.md` / `runpods-5090x4.md` 第 7 节「从零复现」 |
| 要下权重 / 对权重溯源 / 算磁盘预算 | [`models.md`](models.md) |
| 要精确到包版本 | `locks/` 下的 `pip freeze` 与 `conda env export` |
| 要知道**为什么**这么做(实验证据链) | `claude_history/<主题>/FINAL.md`,见 §4 |

> **本文只做对照与决策,不复制细节。** 每个结论后面的路径/命令/字节数都在对应机器档案里;
> 这里给的是「哪台机器是什么定位」「换机器时先看什么」。

## 0. 目录内容

| 文件 | 内容 | 行数量级 |
|---|---|---|
| `README.md` | 本文:三机对照、共性/差异、新机器决策树 | — |
| [`models.md`](models.md) | 跨机权重台账:repo@revision / 体积 / 各机路径 / 校验契约 / 磁盘预算 / 认证 | — |
| [`popos-5090.md`](popos-5090.md) | 2×RTX5090 本地机 | 1200 |
| [`popos-6000a.md`](popos-6000a.md) | 4×RTX6000Ada 本地机 | 880 |
| [`runpods-5090x4.md`](runpods-5090x4.md) | 4×RTX5090 云 pod | 1000 |
| `locks/` | 每机每环境的 `pip freeze` + `conda env export`(共 13 份) | — |

> ⚠️ `locks/` 的**采集口径不统一**:5090 与 runpods 用的是 `pip freeze --all`,6000a 用的是
> `pip freeze`(不带 `--all`)。跨机比对包数前先对齐口径,别拿 108 去减 105。
> 文件头 4–7 行是采集注释,`wc -l` 要减掉。所有 13 份都扫过 token 正则,0 命中。

---

## 1. 三机对照

### 1.1 硬件与拓扑

| | **popos-5090** | **popos-6000a** | **5090-Runpods** |
|---|---|---|---|
| 定位 | 单卡量化产线 + TP2 serving;**stride bug 的根因定位现场** | **全能机**:三条 serving 全跑通,BF16 4 卡 oracle;**权重源头机** | **纯引擎 serving**,提供本地两台给不了的「4×32G + 1007G RAM」 |
| GPU | 2 × RTX 5090,32607 MiB,**sm_120** | 4 × RTX 6000 Ada,49140 MiB,**sm_89** | 4 × RTX 5090,32607 MiB,**sm_120** |
| 互联 | NODE | NODE(**无 NVLink**) | 对内 NODE / 跨对 **SYS** |
| **P2P** | **全 CNS(无)** | **两两全 OK** | **全 CNS(无)** |
| CPU | Threadripper 9960X,24C/48T | Xeon w9-3475X,36C/72T | 2 × Xeon Gold 6530,128 线程 |
| NUMA | 1 | 1 | **2**(GPU0/1→node0,GPU2/3→node1) |
| RAM | **125 GiB**(swap 7G) | 376 GiB(swap 19G) | **1007 GiB**(无 swap) |
| 盘 | 根盘 1.9T,余 1.3T,全部权重同一块 | 根盘 3.7T **94% 满** + 第二块 NVMe 3.7T **93% 满**(大件放这块 + symlink) | `/workspace` 503G 持久卷(余 215G)+ `/` 1T 临时盘(**pod 重建即失**) |
| 特殊占用 | **GPU0 带显示输出**(Xorg/gnome ≈0.5 GiB),而 :8189/:8190 两个 worker 恰在 GPU0 | 共享工作机,GPU 上可能有别人的活 | 独占,但采集时我们自己的服务在跑 |
| 归属 | isaac 的共享工作站 | isaac 的共享工作站 | 租用 pod |

> ⚠️ **`popos-5090` 这台机器不是 Pop!\_OS**(实测 Ubuntu 24.04.4,hostname `Ubuntu-5090`)。
> 别名是历史遗留,写脚本时**不要据别名推断发行版**。
> ⚠️ 5090 上 `/home/isaac/Data`(大写 D)是**另一块盘、另一份 Dropbox**,与项目根
> `/home/isaac/data`(小写 d)无关。6000a 专用的 `download_bf16.sh` 把 `STORE` 写死成前者,
> 在 5090 上**不会报错**,而是把 113G 写进别人的盘。

### 1.2 系统 / 驱动 / wheel 变体(**复现的第一约束**)

| | popos-5090 | popos-6000a | 5090-Runpods |
|---|---|---|---|
| 发行版 | Ubuntu 24.04.4 (noble) | **Pop!_OS 22.04** (jammy) | Ubuntu 24.04.3(容器) |
| 内核 | 7.0.0-28-generic | 7.0.11-76070011-generic | 6.8.0-90-generic(宿主 22.04 HWE) |
| **NVIDIA 驱动** | **580.173.02** | **580.173.02** | **570.195.03** |
| 驱动暴露 CUDA | **13.0** | **13.0** | **12.8** |
| gcc | 13.3.0 | **11.4.0** | 13.3.0 |
| 系统 nvcc | 无 | 无 | 无 |
| **wheel 变体** | `+cu130`(PyPI / `whl/cu130`) | `+cu130` | **`+cu129`,GitHub release 直链** |
| 依据 | CUDA 13.0 运行时要求驱动 ≥580.65,满足 | 同左 | 570 只到 12.8;装 cu130 会 `driver too old (found version 12080)`;**`+cu128` wheel 根本不存在** |

> ⚠️ **不要跨机照抄安装脚本**:`setup_runpods.sh` / `fix_cu129.sh` 的 cu129 绕行方案在两台本地机上
> 完全不需要;反过来,本地机的 `pip install vllm==0.26.0` 在 runpods 上必挂。
> **装之前先 `nvidia-smi`,不要先看文档。**
> ⚠️ 三台都**没有系统 CUDA toolkit**,SGLang 的 JIT 全靠 pip 的 `nvidia-cuda-nvcc` /
> `nvidia-cuda-runtime` 提供 nvcc 与 `libcudart`。**不要装系统 toolkit**(会让 JIT 链到不匹配的库)。

### 1.3 Python 环境

| | popos-5090 | popos-6000a | 5090-Runpods |
|---|---|---|---|
| 管理方式 | conda 26.7.0,**3 个 H3 env**(机上另有 7 个别人的) | conda 26.5.3,**3 个 H3 env**(机上共 18 个) | **无 conda**,单个 `python3 -m venv` |
| 路径 | `~/miniconda3/envs/h3_{comfy,vllm,sglang}_NV_py312` | 同左 | `/workspace/h3/env` |
| Python / pip | 3.12.13 / 26.1.2(三个 env 一致) | 3.12.13 / 26.1.2 | 3.12.3(系统 python)/ 26.2.1 |
| torch | 2.11.0+cu130 | 2.11.0+cu130 | **2.11.0+cu129** |
| 包数(freeze 条目) | 108 / 230 / 239 | 105 / 227 / 236 | **225(一个 env 打全部)** |
| env → 用途 | comfy env 同时跑 **ComfyUI + 全部客户端脚本**;vllm/sglang env 只跑服务端 | 同左 | **1 : 8** —— 模型/精度/拓扑都是 `vllm serve` 的命令行参数,不是环境差异 |
| 为什么分三个 env | 三条产线的 `transformers`/`diffusers`/`numpy`/`pydantic` 互不兼容(comfy 5.14.1+numpy2.4.4;sglang 5.12.1+pydantic 2.14.0b1) | 同左 | 只有 vLLM 一条产线,不需要分 |

> ⚠️ **两台本地机的 sglang env 里 `pip show torch` 报的是裸 `2.11.0`(没有 `+cu130`),但
> `torch.__version__` 三个 env 全是 `2.11.0+cu130`。** 差异只在 pip 元数据层(uv 安装丢了
> local version 标记),**功能等价,不要因此重装 torch**。找通道差异要用 `pip show` / lock 文件,
> 用 `torch.__version__` 永远找不到。
> ⚠️ 数包用 `pip freeze | wc -l`,**不要用 `pip list | wc -l`**(带 2 行表头,会系统性 +2)。

### 1.4 装了哪些后端 / 承担哪些 profile

| | popos-5090 | popos-6000a | 5090-Runpods |
|---|---|---|---|
| ComfyUI | ✅ 3 个 worker(:8188/:8189/:8190) | ✅ 1 个 worker(:8288,四卡可见给 DisTorch2) | ❌ 无 |
| custom_nodes | **TeaCache + Turbo + MultiGPU**(3 个) | **只有 MultiGPU** | — |
| vLLM-Omni | ✅ :8091,TP2 + DLO | ✅ :8091,TP4,**不开 DLO** | ✅ :8091,TP4 / TP2×U2,DLO |
| SGLang | ⚠️ env 在,但 editable 指向**已判死**的 PR33681 分支 | ✅ :30010,TP4 FP8(768P) | ❌ 未安装 |
| **profile 数** | **5** | **8** | **8** |
| profile 名 | `comfy-int8-{original,teacache,turbo}-1c`<br>`vllm-fp8-{original,turbo}-tp2` | `comfy-int8-original-1c`,`comfy-bf16-original-4c`<br>`vllm-{bf16,fp8}-{original,turbo}-tp4`<br>`sglang-fp8-{original,turbo}-tp4` | `vllm-{bf16,fp8}-{original,turbo}-{tp4,tp2u2}` |
| 切换入口 | `h3_switch_5090.sh` | `h3_switch.sh` | `h3_switch_runpods.sh <model> <prec> <topo> [resident]` |

> ⚠️ **三份 switch 脚本的 CLI 契约互不兼容**(共 446 行),合并成一份参数化版是 Phase 2 的事。
> ⚠️ `h3_generate.py` 的 runpods 旧名别名 **`turbo-lora → vllm-bf16-original-tp4` 是错的**
> (把 turbo 映射到了 original)。修掉之前**只用全名**。

### 1.5 最快产线与延迟

统一素材 ToS 首尾帧,864×480 / 124 帧(5.167s @24fps)/ 带音频,**warm 中位**:

| 机器 | 最快 profile | NFE4 | NFE6 | 占卡 | 峰值显存/卡 |
|---|---|---|---|---|---|
| **5090-Runpods** | `vllm-fp8-turbo-tp2u2` | **16.3s** | **18.5s** | 4 | 19.2G |
| popos-6000a | `vllm-fp8-turbo-tp4` | 17.9s | 23.5s | 4 | ~38G |
| popos-5090 | `vllm-fp8-turbo-tp2` | 22.0s | 29.0s | 2 | ~24G |
| popos-5090 | `comfy-int8-turbo-1c` | 25.0s | **30.0s** | **1** | ~31G |

**同机内的横向对照**(说明「最快」是相对什么):

| 机器 | profile | 步数 | warm | 近似手段 |
|---|---|---|---|---|
| 5090 | `comfy-int8-original-1c` | 30 步 | ~95–110s | 无 |
| 5090 | `comfy-int8-teacache-1c` | 12 步(~7 实算) | 35.0s | TeaCache 有损缓存 |
| 5090 | `vllm-fp8-original-tp2` | NFE11 | 46.9s | 无(在线 FP8) |
| 6000a | `comfy-int8-original-1c` | 30 步 | ~100s 热 / 210s 冷 | 无 |
| 6000a | `comfy-bf16-original-4c` | 30 步 | 400s | **零量化,质量参照** |
| 6000a | `vllm-fp8-original-tp4` | NFE11 | 37.4s | 在线 FP8 |
| 6000a | `vllm-bf16-original-tp4` | NFE11 | 44.2s | **无损** |
| 6000a | `sglang-fp8-original-tp4` | NFE11 | 126.2s | **1376×768 高清档**(像素 4.3×,不可与 864×480 直比) |

要点:
- **5090 单卡 ComfyUI Turbo(30.0s / 1 卡)与自己的 TP2 引擎(29.0s / 2 卡)打平** ——
  两张卡各起一个 turbo worker 换 ~2× 吞吐,是引擎 TP 路线给不了的(该并行配置**尚未实测**)。
- **runpods 上 TP2×U2 快过 TP4**(18.5 vs 20.4s):无 P2P 时 ring all-reduce 的通信量随 N 上升,
  而 Ulysses 在 attention 处只做 all-to-all。**拓扑决定并行策略,不是「卡越多越快」。**
- **SGLang 只能做 768P**:公共 API 源码硬锁 `short_edge=768`,低延迟小画布在该 commit 上走不通。

> ⚠️ **所有数字必须声明冷热态**:`/health` 在两个引擎上都**早于权重加载就绪**返回 200,
> 切换后第一单必然是 warmup,不计时。冷/热差距实测:5090 ComfyUI 45.0 vs 25.0s;
> runpods 冷启到 `/health` 200 约 175–180s、warmup 23.1 vs 稳态 18.5s。
> (那对 45.0/25.0 是 `comfy-int8-turbo-1c` @ NFE4,**不是** 30 步基线 —— 基线本来就 ~95–110s。)
> 计时协议 = 固定 seed + warmup 1 不计 + timed 3 取中位(`scripts/h3_eval.py`);
> **ComfyUI 另需换 seed/prompt 破整图缓存**,引擎侧不需要。

### 1.6 权重占盘

| | popos-5090 | popos-6000a | 5090-Runpods |
|---|---|---|---|
| 官方 BF16 基座 | 145G(FL2VA 135G + **11G 残留**) | 145G(FL2VA 135G + **11G 残留**) | 135G(+ **34G 残留**) |
| merged Turbo | 62G | 62G | 62G |
| ComfyUI 单文件 | 85G(6 个) | 51G(INT8 组 + 2 VAE,**在根盘**) | — |
| BF16 单文件(L2 oracle) | ❌ 没有 | **118G**(66.3G DiT + 51.5G TE) | ❌ 没有 |
| LoRA | 0.74G(1 个) | 1.5G(2 个) | 0.74G(1 个) |
| 死重/残留 | **~50G**(TE FP8 34G + modelopt 5G + cache 11G) | 11G | **56G**(base 34G + LoRA 22G) |
| **生产必需** | **~283G** | **~376G** | **197G** |
| **当前占用** | ~332G | ~376G(两块盘都 93–94% 满) | 263G(含 venv 12G) |

> ⚠️ **本表的 G = GiB(`du -sh` 口径)**,而 `models.md` §1 报的是精确字节数换算出的 **GB**。
> 同一个文件两种写法都会出现,别当成两份不同的权重:
> Turbo LoRA `779 849 816 B` = **0.74 GiB** = 0.78 GB(runpods 档案里写的 780M 就是后者);
> BF16 单文件 DiT `66 280 487 368 B` = **62 GiB** = 66.3 GB(models.md §4.1 的「+62G」与
> §1 的「66.3G」是同一个文件)。

明细、字节数、校验契约、下载命令全部在 [`models.md`](models.md)。

---

## 2. 共性 vs 差异

**换一台新机器时,先读这一节。** 左边那栏改一处就要改三台,右边那栏每台机器自己算。

### 2.1 三机必须一致(改一台 = 改全部,否则 benchmark 不可比 / 输出不对)

| # | 约束 | 判据 / 出处 | 不一致的后果 |
|---|---|---|---|
| 1 | **stride 补丁**(`pr5910_resident_stride_fix.patch`) | `sed -n '/class PinnedResidentLayerGroup/,/def offload/p' <dlo.py> \| grep -q as_strided` | 凡走 vLLM DLO 的机器,不打就**静默产纯噪声**或崩 |
| 2 | **判定补丁只看代码内容** | 同上 | `git diff` 在补丁已 commit 时会反报「没打」;**整文件 `grep as_strided` 恒真**(原文件 299/412 行本来就有两处) |
| 3 | **NFE 语义** | 引擎 `num_inference_steps = NFE + 1`;ComfyUI `steps == forwards`;客户端 `--nfe` 统一 | 直接错一步,两边数字不可比。升级引擎后**必须重跑 `scripts/test_h3_schedule.py`**(上游有开放 PR 要改语义) |
| 4 | **输出契约** | 帧数 `n % 17 == 5`(124 帧 = 5.167s @24fps);画布 32 的倍数;AAC 立体声 32kHz | 帧数不合法直接失败 |
| 5 | **audio VAE 必须 fp32** | 见 `models.md` | BF16 会让音量 **−20 dB**,而且不报错 |
| 6 | **采样配方** | video shift 12 / audio shift 3(引擎默认即 canonical);LoRA strength 1.0;alpha 缺省 = rank(scale 1.0) | 输出不可比 |
| 7 | **merged checkpoint 契约** | 根目录 `.complete` + `merge_manifest.json` + `delta_norms.csv`;`259 / 535`;`lora_sha256`;`base_transformer_index_sha256` | switch 脚本靠 `.complete` 拒绝启动半成品 |
| 8 | **vLLM 只走 merged checkpoint** | runtime LoRA 对 fused 层(qkv 21504 行 / fc1 28672 行)**warning 后跳过** | 蒸馏**静默失效**;dynamic LoRA 只在 SGLang 做,且必须 `--lora-merge-mode dynamic`(`auto` 会把 LoRA merge 进 FP8 权重) |
| 9 | **不能用 ComfyUI 核心 `LoraLoaderModelOnly`** | `comfy/lora.py` 的 `model_lora_keys_unet` 没有 MiniMaxH3 分支 | 裸键 LoRA **0/518 命中且不报错**,出片 = 「低步数的 base」。必须用 `MiniMaxH3TurboLoRA` 作者节点 |
| 10 | **benchmark 协议** | warmup 1 不计 + timed 3 取中位,固定 seed;必须声明冷热态 | `/health` 早于权重就绪,首单必是 warmup |
| 11 | **HF 下载姿势** | `HF_XET_HIGH_PERFORMANCE=1`(**不是**已弃用的 `HF_HUB_ENABLE_HF_TRANSFER`);`--include` 必须重复写;LoRA 必须 `--include` 单文件 | 详见 `models.md` §5 |
| 12 | **ssh 铁律** | 远端路径用单引号或 `\$HOME`;`pgrep`/`pkill` 模式用方括号 | 双引号里的 `~` 会被 **Mac 本地** shell 展开成 `/Users/ning` |
| 13 | **项目根 = git 工作区** | 三机采集时同为 `main @ 39c4b508`,worktree 干净 | `git pull` 就是更新代码 |

### 2.2 每台机器特有(必须逐机决定的)

| 维度 | popos-5090 | popos-6000a | 5090-Runpods |
|---|---|---|---|
| **wheel 变体** | `+cu130` | `+cu130` | **`+cu129`(GitHub 直链)** |
| **env 管理** | conda × 3 | conda × 3 | 单 venv |
| **项目根** | `~/data/dropbox/CV/h3` | `~/data/dropbox/CV/h3` | `/workspace/h3` |
| **权重布局** | 全在根盘(1.3T 余量) | 大件在**第二块 NVMe** + symlink 回 `ComfyUI/models` | 全在 `/workspace` 持久卷 |
| **attention backend** | `CUDNN_ATTN` | **`FLASH_ATTN`** | `CUDNN_ATTN` |
| **`VLLM_DISABLED_KERNELS`** | ✅ 禁 `CutlassFP8ScaledMMLinearKernel`(sm_120) | ❌ 不设(sm_89 上 Cutlass 可用) | ✅ 禁 |
| **DLO** | ✅ `--dlo-resident-layers 50`(H3 恰 50 层 = 全常驻) | ❌ 不开 | ✅ bf16→40 / fp8→50 |
| **`PYTORCH_CUDA_ALLOC_CONF`** | 仅实验脚本设 | 仅实验脚本设 | ✅ 常规 switch 就设 `expandable_segments:True` |
| **stride 补丁存放方式** | 分支 `pr5910-stride-fix`(**生产 env 就指向它**) | 分支 `pr5910-stride-fix`,但那是**影子 checkout**(生产走 `src/vllm-omni` main 且不开 DLO,走不到常驻路径) | ⚠️ **未提交的工作区修改** —— `git checkout/reset/stash/切分支` 都会静默抹掉它,而引擎不报错、只开始产噪声 |
| **custom_nodes** | TeaCache + Turbo + MultiGPU | 只有 MultiGPU | 无 ComfyUI |
| **HF cache 桥接** | 有(为 SGLang) | 有(为 SGLang) | 无(没装 SGLang) |
| **`_ASYNC_OUTPUT_TIMEOUT`** | 未打(30.0) | 未打(30.0) | 未打(30.0)——**三机都没打,档案里的「必打」是建议不是现状** |

> ⚠️ **`launch_comfy.sh`(5090 :8188)没设 `CUDA_DEVICE_ORDER`**,另两个 launcher 设了 `PCI_BUS_ID`。
> 两卡同型号时通常等价,但不是保证 —— 出现「:8188 跑到了显示卡上」先查这个。
> ⚠️ **`ComfyUI-MultiGPU` 即使跨卡路线已放弃也仍在每个 worker 里 monkey-patch 核心**
> (`mm.get_torch_device` / `comfy.sample.sample` / comfy_kitchen DLPack guard)。
> 所有历史 ComfyUI benchmark 都是在这个前提下测的:**要么保留(推荐),要么移除后全部重测。**

### 2.3 三处已知的「脚本骗人」(跨机通用,排障时先排除)

| 现象 | 真相 | 正确判据 |
|---|---|---|
| 日志抬头 `stride_patch=APPLIED_as_strided` | **恒真、无鉴别力**(`h3_switch_5090.sh`、`h3_switch_runpods.sh:106`、`setup_runpods.sh:43` 用的都是裸整文件 grep) | 手工跑类内限定的 `sed … \| grep -q as_strided`(`post_setup2.sh` 里那段是对的) |
| `merge_manifest.json` 的 `verification` 字段写着 `L2 Comfy single-file layout oracle` | **静态模板**;5090 与 runpods 的 L2 实际被**静默跳过**(路径不存在时 `merge_turbo_lora.py:264` 直接走 else,退出码仍是 0) | 只看 `logs/merge*.log` 里有没有 `verify L2 passed` |
| `pip show <editable 包>` 的版本号 | 停在**安装时**的快照(6000a sglang 停在 `g407a65d3c`,实际 HEAD 是 `4c28e24`) | `git rev-parse` / `pip freeze` |

---

## 3. 按硬件选型的决策树

> 全部基于三机实测。**任何新机器,先跑完 §3.1 的五条命令再决定装什么。**

### 3.1 五个必测底数(装任何东西之前)

```bash
nvidia-smi --query-gpu=driver_version,name,memory.total,compute_cap --format=csv   # ① 驱动 ② 显存 ③ 架构
nvidia-smi topo -p2p r                                                             # ④ P2P
nvidia-smi topo -m; lscpu | grep -i numa                                           # ④' 卡落在哪个 NUMA
free -g                                                                            # ⑤ RAM —— serving 可行性的硬门槛
df -h <权重盘>                                                                      # 磁盘预算见 models.md §4
which ffmpeg ffprobe flock ss                                                      # 硬依赖,缺了整条 probe/bench 链路静默失效
```

### 3.2 决策树

```text
① 驱动版本?
   ├─ ≥ 580.65(暴露 CUDA 13.0)
   │     → torch 2.11.0+cu130(download.pytorch.org/whl/cu130)+ PyPI vllm==0.26.0
   │       实证:popos-5090 / popos-6000a
   └─ 570.x(只到 CUDA 12.8)
         → 必须 vllm-0.26.0+cu129 GitHub release 直链 + --extra-index-url .../cu129
           实证:5090-Runpods。装 cu130 会在 torch.cuda.init() 抛
           "The NVIDIA driver on your system is too old (found version 12080)"
           ⚠️ +cu128 wheel **不存在**,官方文档那句是过期文本,别去找
   (三种情况都不装系统 CUDA toolkit;SGLang 的 nvcc/libcudart 全靠 pip 的 nvidia-cuda-*,
    且 CUDA_HOME 硬写死到 site-packages/nvidia/cu13 → 这些包必须钉版本,
    落到 nvidia/cu14/ 会让四个路径变量全指空)

② compute capability?
   ├─ sm_89(Ada / 6000a)
   │     → Cutlass FP8 可用,不设 VLLM_DISABLED_KERNELS
   │       attention 用 FLASH_ATTN
   └─ sm_120(Blackwell / 5090)
         → VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel
           attention 用 CUDNN_ATTN(实测 TORCH_SDPA 慢约 18s:46.8 → 64.8s)
         ⚠️ 「sm_120 上 FP8 全线不可用」这个 round-1 结论**已被推翻**:
            崩(严格 kernel 拒收)与噪声(宽松 kernel 照算乱序字节)是 stride bug 的
            两种表象,与架构无关。见 claude_history/08_5090_tp2/FINAL.md

③ 每卡显存?(DiT 权重口径,不含激活/TE/VAE)
   ├─ 48G(6000a)→ BF16 TP4 全常驻可行(实测峰值 ~46G/卡);不需要 DLO
   │              ⚠️ 但 768P + TP4 BF16 会 OOM → SGLang 产线固化 --quantization fp8
   └─ 32G(5090)→ BF16 66G:TP4 16.5G ✓ / TP2 33G ✗(超 31.3G 可用)
                  FP8 33G:TP4 8.3G ✓ / TP2 16.5G ✓
                  ⇒ BF16 想上 TP2 必须开 DLO;实测 bf16+tp2u2+r40 峰值 28430 MiB,
                    余量仅约 4G,往上调 resident 有 OOM 风险
                  ⚠️ 纸面估算会误判:按「BF16 TP2 每卡 33G」判「装不下」,开了 DLO 实测 28.4G

④ 主机 RAM?  ← **serving 可行性的真门槛,不是显存**
   ├─ ≥ ~376G(6000a 376G / runpods 1007G)
   │     → 常规加载路径可行;SGLang cookbook 的 2×5090 配方明确要求 "377 GiB host"
   ├─ ~125G(popos-5090)
   │     → **常规 serving 结构性不可行**(2026-08-08 实测判死):
   │        · vLLM TE-TP + offload:运行时 48G/卡 > 31.3G,VAE init 时 OOM
   │        · vLLM 常规加载:H3 不支持 mmap → 全量权重读进 RAM → 125G 打爆(rank0 被内核杀)
   │        · SGLang TP2:两个 rank **各自**暂存整份 TE(2×51.5G=103G)→ RAM 打爆
   │     → **唯一活路 = PR5910 的 DLO 路线**(2026-08-09 修复上线):
   │        TP2 + 在线 FP8 + --enable-distributed-layerwise-offload --dlo-no-use-allgather
   │        --dlo-resident-layers 50,实测 base NFE11 46.9s / turbo NFE6 29.0s
   │     ⚠️ 走 DLO ⇒ 常驻组路径 100% 命中 ⇒ **stride 补丁从「建议」升级为「必打」**
   └─ < 125G → 不要考虑 serving,走 ComfyUI 单卡量化产线

⑤ P2P?
   ├─ 两两全 OK(6000a)
   │     → TP4 直接上;ComfyUI DisTorch2 跨卡分置也可用(BF16 4 卡 oracle 的物理前提)
   └─ 全 CNS(5090 / runpods)
         → 所有 NCCL 集合走 host 中转
         → TP 分组必须落**同 NUMA**(runpods:(0,1) 与 (2,3))
         → **TP4 不必然最快**:ring all-reduce 每卡流量 ∝ 2(N−1)/N,N 从 2→4 通信量 ×1.5,
           而每卡算力只减半;Ulysses 在 attention 处只做 all-to-all
           实测 runpods:tp2u2 18.5s < tp4 20.4s(NFE6 turbo FP8)
         → ⚠️ tp2u2 时 --text-encoder-tp-size 必须 = 世界大小(4),传 2 会在
           _build_text_encoder_group 的 assert cpu_group is not None 处全崩
         → ⚠️ **ComfyUI-MultiGPU 跨卡在 comfy-kitchen 环境下必崩**(CPU 中转与 dlpack 不兼容,
           illegal memory access)。5090 上已判死,**别再试**

⑥ 有没有 ComfyUI BF16 单文件权重?
   ├─ 有(6000a)→ merge 时给 --comfy-single,**L2 独立布局 oracle 真跑**
   └─ 没有       → L2 会被**静默跳过**(退出码仍 0),只能靠与 6000a 的溯源哈希比对替代
                  详见 models.md §3
```

### 3.3 一页纸决策表

| 观测 | 结论 | 实证机器 |
|---|---|---|
| 驱动 ≥580.65 | `+cu130` wheel | 5090 / 6000a |
| 驱动 570 / CUDA 12.8 | `+cu129` GitHub 直链;**无 cu128** | runpods |
| sm_120 | 禁 Cutlass FP8 kernel + `CUDNN_ATTN` | 5090 / runpods |
| sm_89 | Cutlass 可用 + `FLASH_ATTN` | 6000a |
| 48G/卡 | BF16 TP4 可全常驻,不必 DLO;768P BF16 仍 OOM | 6000a |
| 32G/卡 | BF16 TP2 必须 DLO;FP8 两种拓扑都放得下 | 5090 / runpods |
| RAM ≥376G | 常规 serving 可行 | 6000a / runpods |
| RAM ~125G | 常规 serving 判死;**只有 DLO 路线活** | 5090 |
| P2P 全 OK | TP4 优先,DisTorch2 可用 | 6000a |
| P2P 全 CNS + 2 NUMA | TP 分组同 NUMA;**先测 tp2u2 再测 tp4** | runpods |
| P2P 全 CNS + 2 卡 | 跨卡 ComfyUI 别试;引擎走 TP2+DLO | 5090 |
| 没有 Comfy BF16 单文件 | merge 的 L2 会静默跳过,必须另找证据链 | 5090 / runpods |

---

## 4. 深度链接

**实验证据链**(每个主题一份权威 `FINAL.md`,原始记录在同目录的 session 子目录):

| 主题 | 一句话 |
|---|---|
| [`claude_history/06_infra/FINAL.md`](../../claude_history/06_infra/FINAL.md) | **装机总入口**:机器/端口/env/版本 pin/必打补丁的台账 |
| [`claude_history/01_deploy_smoke/FINAL.md`](../../claude_history/01_deploy_smoke/FINAL.md) | 双机部署与冒烟:env、ComfyUI pin、Comfy 单文件权重清单 |
| [`claude_history/02_5090_turbo/FINAL.md`](../../claude_history/02_5090_turbo/FINAL.md) | 5090 TeaCache 产线的生产配置(thresh 0.10 / start 2 / end −2) |
| [`claude_history/03_6000a_bf16_oracle/FINAL.md`](../../claude_history/03_6000a_bf16_oracle/FINAL.md) | 4 卡 BF16 DisTorch2 质量参照(400s,零量化) |
| [`claude_history/04_6000a_serving/FINAL.md`](../../claude_history/04_6000a_serving/FINAL.md) | 6000a 三条 serving 产线定型与分工 |
| [`claude_history/05_5090_serving/FINAL.md`](../../claude_history/05_5090_serving/FINAL.md) | **125G RAM 判死常规 serving 的完整根因**(决策树 ④ 的出处) |
| [`claude_history/07_turbo_lora/FINAL.md`](../../claude_history/07_turbo_lora/FINAL.md) | Turbo LoRA merge、三层校验(L1/L2/L3)、NFE 语义定案 |
| [`claude_history/08_5090_tp2/FINAL.md`](../../claude_history/08_5090_tp2/FINAL.md) | **stride bug 根因与 TP2 修复上线**;推翻「sm_120 FP8 不可用」 |
| [`claude_history/09_runpods_tp4/FINAL.md`](../../claude_history/09_runpods_tp4/FINAL.md) | 4×5090 云机 TP4 vs TP2×Ulysses2,无 P2P 下的拓扑结论 |
| [`claude_history/10_repo/FINAL.md`](../../claude_history/10_repo/FINAL.md) | 仓库化与跨机同步(项目根 = git 工作区) |

**benchmark 全表与方案分析**(`doc/`):

| 文件 | 内容 |
|---|---|
| `doc/high_level.md` | 项目总体方案 |
| `doc/speedup_5090_results.md` | 5090 ComfyUI TeaCache 矩阵(B1–B5 + FP8 A/B) |
| `doc/speedup_5090_serving_results.md` | 5090 serving 判死 + **TP2 攻坚 round-2 证据链** |
| `doc/speedup_6000a_results.md` | 6000a 三产线总表 |
| `doc/speedup_runpods_tp4_results.md` | runpods 拓扑矩阵与部署踩坑 |
| `doc/turbo_lora_results.md` | Turbo LoRA 全表(三机)+ checkpoint 溯源 PSNR 复核 |
| `doc/TP2_5090.md` / `doc/TP_5090_round2.md` / `doc/TP4_5090.md` | 三份 research note(执行前的方案推演) |

---

## 5. 采集与维护

- 三份档案的采集日期均为 **2026-08-09**,全程**只读**:未修改任何机器上的文件、未装卸包、
  未启停服务、未跑生成任务。唯一写盘动作是把清单落到本仓库的 `locks/`。
- **重新采集**:见各机档案的第 9 节「本文档的采集方式」,那里列了可直接粘贴的 ssh 命令组。
- 重采时最容易记错的三个口径(都在上一轮复核里踩过):
  1. 包数用 `pip freeze | wc -l`,不是 `pip list | wc -l`;三机 lock 的 `--all` 口径不统一。
  2. torch 通道差异用 `pip show torch`,不是 `torch.__version__`。
  3. `du -sh` 的体积里含下载残留,报数时必须拆解(5090 的 145G 含 11G、runpods 的 168G 含 34G)。
- 本文与 `models.md` 是**派生文档**:三份机器档案更新后,回来核对 §1 的表格与 §2.2 的差异列。
