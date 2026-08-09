# 机器档案:5090-Runpods(RunPods 4×RTX 5090 云 pod)

> 采集日期 **2026-08-09**(pod 时区 UTC),全部事实来自当天在该机实跑的只读命令。
> 凡属推断的地方都标了"(推测,依据:…)"。与历史档案冲突处以实测为准并在正文点明。
> 重新采集方式见第 9 节。

---

## 1. 这台机器是什么

**一台只跑 vLLM-Omni 引擎侧 serving 的租用 GPU pod**:4×RTX 5090、双路 Xeon、1007G 内存,
没有 conda、没有 ComfyUI、没有 SGLang,整个产线由一个 venv(`/workspace/h3/env`)+ 一份
vllm-omni 源码(`/workspace/h3/src-vllm-omni`)撑起来。它存在的理由是**本地两台机器都给不了的
两个条件**:4 张 32G 卡 + 1007G 主机内存(popos-5090 只有 125G,serving 路线在那里结构性跑不动)。

### 硬件(实测)

| 项 | 值 | 采集命令 |
|---|---|---|
| hostname | `b35d8f1afb59`(容器) | `hostname` |
| GPU | 4 × NVIDIA GeForce RTX 5090,**32607 MiB/卡**,compute capability **12.0(sm_120)** | `nvidia-smi --query-gpu=...` |
| GPU 功耗墙 | 575.00 W/卡,最大 SM 时钟 3090 MHz,persistence mode **Enabled** | `nvidia-smi --query-gpu=power.limit,clocks.max.sm,persistence_mode` |
| PCIe | 链路能力 **Gen5 ×16**;空闲时 current = Gen1 ×16(省电降速,不是故障) | `nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.gen.current` |
| CPU | 2 × **INTEL(R) XEON(R) GOLD 6530**,32 核/路 × 2 线程 = **128 逻辑核**,base 0.8–4.0 GHz | `lscpu` |
| NUMA | **2 个节点**:node0 = CPU 0-31,64-95;node1 = CPU 32-63,96-127 | `lscpu` |
| 内存 | **1007 GiB** total(采集时 used 193 / free 324 / buff-cache 639 / available 813),**无 swap** | `free -g` |
| /dev/shm | **268G**(vLLM-Omni worker 之间靠共享内存传请求,日志 `ready to receive requests via shared memory`) | `df -h /dev/shm` |
| 持久卷 | `/workspace` = `/dev/mapper/pod-ocqy0ibumlsdvk`,**ext4,503G,已用 263G,可用 215G(56%)** | `df -h /workspace` |
| 临时盘 | `/` = docker overlay,1.0T,已用 5.7G。**pod 重建即失** | `df -h /` |
| 网卡 | NIC0/NIC1 = `roceP1p148s0f0` / `roceP1p148s0f1`(RoCE),挂在 **NUMA1** | `nvidia-smi topo -m` |

### GPU 拓扑(这是本机所有并行策略的根因)

```
        GPU0   GPU1   GPU2   GPU3   NIC0   NIC1   CPU Affinity     NUMA
GPU0     X     NODE   SYS    SYS    SYS    SYS    0-31,64-95        0
GPU1    NODE    X     SYS    SYS    SYS    SYS    0-31,64-95        0
GPU2    SYS    SYS     X     NODE   NODE   NODE   32-63,96-127      1
GPU3    SYS    SYS    NODE    X     NODE   NODE   32-63,96-127      1
```

`nvidia-smi topo -p2p r` 与 `-p2p w` **两张表全部是 `CNS`(Chipset not supported)**——
四张卡两两之间**没有任何 P2P**,每一次 NCCL 集合通信都要经主机内存中转。

两条直接后果:

1. **TP2 分组必须落 (0,1) 和 (2,3)** —— 同 NUMA 节点内中转,跨 NUMA 走 UPI 会再慢一档。
2. **TP4 不必然最快**。Ring all-reduce 每卡流量 ~2(N−1)/N·S,N 从 2 涨到 4 时通信量上升;
   而 Ulysses(序列并行)在 attention 处只做 all-to-all。实测 `tp2u2`(TP2 × Ulysses2)
   在 Turbo FP8 上比 TP4 快:**18.5s vs 20.4s**(NFE6,中位数,见 6/8 节数据表)。
   > ⚠️ 拓扑决定并行策略。任何新硬件上排实验顺序之前先跑 `nvidia-smi topo -p2p r`。

### 它在项目里承担哪些 profile

`scripts/h3_generate.py --list` 里归本机的 8 个(全部是 serving,ComfyUI 一个都没有):

| profile | switch 参数 | NFE | 画布 |
|---|---|---|---|
| `vllm-bf16-original-tp4` | `original bf16 tp4` | 11 | 864×480 |
| `vllm-fp8-original-tp4` | `original fp8 tp4` | 11 | 864×480 |
| `vllm-bf16-original-tp2u2` | `original bf16 tp2u2` | 11 | 864×480 |
| `vllm-fp8-original-tp2u2` | `original fp8 tp2u2` | 11 | 864×480 |
| `vllm-bf16-turbo-tp4` | `turbo bf16 tp4` | 6 | 864×480 |
| `vllm-fp8-turbo-tp4` | `turbo fp8 tp4` | 6 | 864×480 |
| `vllm-bf16-turbo-tp2u2` | `turbo bf16 tp2u2` | 6 | 864×480 |
| `vllm-fp8-turbo-tp2u2` | `turbo fp8 tp2u2` | 6 | 864×480 |

三个维度(模型 / 精度 / 拓扑)正交,统一入口:

```
bash /workspace/h3/scripts/h3_switch_runpods.sh <original|turbo> <bf16|fp8> <tp4|tp2u2> [resident]
bash /workspace/h3/scripts/h3_switch_runpods.sh stop
```

> ⚠️ **`h3_generate.py` 的 runpods 旧名别名有 bug(已实测确认)**:
> `LEGACY_ALIASES["runpods"]["turbo-lora"] = "vllm-bf16-original-tp4"` ——
> 把一个 **turbo** 旧名映射到了 **original** profile。`--list` 输出里明晃晃写着
> `runpods: turbo-lora→vllm-bf16-original-tp4`。用 `--profile turbo-lora` 打 runpods
> 会静默拿到基座模型 + NFE11,而不是 Turbo + NFE6。**在修掉之前只用全名。**

### 采集时的现场状态

采集当下机器**不空闲**:一个我们自己的 vLLM 服务正在跑(pgid 283690,已运行 1h56m),
`run/vllm.variant = turbo-fp8-tp4-r50`,`/health` 返回 200。四个
`vLLM-Omni::DiffusionWorker-{0..3}` 各占 2396–3036 MiB(DLO 常驻权重此刻不在卡上)。
本次盘点全程只读,没有停/起任何服务。

---

## 2. 系统与驱动

| 项 | 值 |
|---|---|
| 发行版 | **Ubuntu 24.04.3 LTS (Noble Numbat)** |
| 内核 | **6.8.0-90-generic**,版本串 `#91~22.04.1-Ubuntu`(宿主提供的 22.04 HWE 内核跑 24.04 用户态,容器场景正常) |
| NVIDIA 驱动 | **570.195.03** |
| 驱动宣称的 CUDA | **12.8** |
| gcc / g++ | **13.3.0**(Ubuntu 13.3.0-6ubuntu2~24.04) |
| GNU ld | 2.42 |
| 系统 nvcc | **没有**(`which nvcc` 空)。CUDA 工具链全部来自 pip 的 `nvidia-*-cu12` 轮子 |
| 系统 python | `/usr/bin/python3` = **3.12.3** |
| conda | **不存在**(PATH 里没有,`/opt/conda`、`/root/miniconda3`、`/root/anaconda3`、`/workspace/miniconda3` 全部不存在) |

### 驱动版本决定了哪些 wheel 变体 —— 本机最关键的一条约束

驱动 570.195.03 只到 **CUDA 12.8**。CUDA 的兼容规则:

- **CUDA 12.x 家族内 minor-version compatibility**:12.9 编译的二进制可以在 12.8 驱动上跑
  (12.x 要求 driver ≥525);
- **CUDA 13.0 要求 driver ≥580.65**。

所以:

| wheel 变体 | 本机能不能用 | 依据 |
|---|---|---|
| `torch 2.11.0+cu130` / PyPI 上的 `vllm==0.26.0` | ❌ **启动即死**:`RuntimeError: The NVIDIA driver on your system is too old (found version 12080)` | 装机日志 + `scripts/fix_cu129.sh` 头注释 |
| `vllm-0.26.0+cu129`(GitHub release 直链)+ `torch 2.11.0+cu129` | ✅ **生产在用**,实测 `torch.cuda.is_available() == True`,4 卡可见 | 本次实测 |
| `+cu128` wheel | —— **根本不存在**。官方文档那句 "also provide CUDA 12.8 binaries" 是过期文本,别去找 | 历史档案,本次未复核上游 |

> ⚠️ 如果换了 pod、驱动 ≥580(比如本地 popos-5090 是 580.173.02),就**不需要** cu129 特供轮子,
> 普通 `pip install vllm==0.26.0` 即可。**装之前先 `nvidia-smi` 看驱动**,不要照抄本文档的 wheel URL。

torch 实测能看到的东西:

```
torch 2.11.0+cu129   torch.version.cuda 12.9   cudnn 91701
cuda.is_available True   device_count 4
gpu0..3  NVIDIA GeForce RTX 5090  sm_120  31.37 GiB
arch list ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120','compute_120']
```

sm_120 在 arch list 里 —— 这一步必须验,否则 5090 会退到 PTX JIT 或直接不可用。

---

## 3. Python 环境

**本机只有一个 Python 环境。** `find /workspace /root /opt -maxdepth 4 -name pyvenv.cfg` 只返回
`/workspace/h3/env/pyvenv.cfg` 一条;没有 conda 因此也没有 conda env。

> 与历史档案的冲突 ①:档案的现场布局里写着还有一个 `env_cu130_unusable/`(旧 venv,约 15G)。
> **实测已不存在** —— 已经被清掉了,215G 可用空间里包含这部分。

### 3.1 `/workspace/h3/env` —— 唯一环境,服务全部 8 条 runpods 产线

| 项 | 值 |
|---|---|
| 路径 | `/workspace/h3/env` |
| 类型 | `python3 -m venv`(`pyvenv.cfg`:`command = /usr/bin/python3 -m venv /workspace/h3/env`) |
| Python | **3.12.3**,`executable = /usr/bin/python3.12` |
| pip | **26.2.1** |
| 体积 | 12G |
| 包总数 | 225 条 `pip freeze` |
| 谁在用它 | `scripts/h3_generate.py` 的 `HOSTS["runpods"]["remote_python"] = "/workspace/h3/env/bin/python"`;`h3_switch_runpods.sh` 的 `VENV="$B/env/bin"`,启动的是 `$VENV/vllm`;`probe.sh` / `bench_matrix.sh` 也都调 `$B/env/bin/python` |
| **环境 → 用途 对应关系** | **1 : 8** —— 这一个 venv 同时服务 original/turbo × bf16/fp8 × tp4/tp2u2 全部组合。模型与拓扑是 `vllm serve` 的**命令行参数**,不是环境差异,所以不需要多环境 |

本机**没有 ComfyUI 相关脚本会调的解释器**:`launch_comfy*.sh` 虽然因为"项目根 = git 工作区"
而躺在 `scripts/` 里,但它们引用的是两台本地机的 conda 路径,在本机根本跑不起来,也从未被调用。

### 3.2 关键包版本与来源(实测)

| 包 | 版本 | 来源 / 备注 |
|---|---|---|
| **torch** | `2.11.0+cu129` | PyTorch cu129 index |
| **torchvision** | `0.26.0+cu129` | 同上 |
| **torchaudio** | `2.11.0+cu129` | 同上 |
| torchcodec | `0.15.0+cu129` | vllm 依赖 |
| **vllm** | `0.26.0+cu129` | **GitHub release 直链 wheel**,freeze 里记录为 `vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=6ce4ca30616f0a35810391015622b197a7b8b267ed27f8716f0789db79ff578b` |
| **vllm-omni** | `0.26.0` | **editable**。`pip show` → `Editable project location: /workspace/h3/src-vllm-omni`;freeze 记为 `-e git+https://github.com/vllm-project/vllm-omni.git@1a9b9c2c13c97763567405f43c5fee994c43ab13#egg=vllm_omni`;`import vllm_omni` 解析到 `/workspace/h3/src-vllm-omni/vllm_omni/__init__.py` |
| **sglang** | **未安装** | `import` ABSENT,`pip show` NOT INSTALLED |
| **flash-attn** | **未安装** | 同上。attention 走 `CUDNN_ATTN`,不需要 |
| **sageattention** | **未安装** | 同上 |
| **xformers** | **未安装** | 同上 |
| transformers | `5.14.1` | PyPI |
| diffusers | `0.38.0` | PyPI(vllm-omni 依赖) |
| accelerate | `1.12.0` | PyPI |
| safetensors | `0.8.0` | PyPI |
| numpy | `2.3.5` | PyPI |
| **triton** | `3.6.0` | torch 依赖 |
| huggingface_hub | `1.27.0`(+ `hf-xet 1.6.0`) | PyPI |
| aenum | `3.1.16` | **注意**:装 vllm-omni 时被从 3.1.17 **降级**到 3.1.16(见 `logs/install_cu129.log`) |
| setuptools-scm | `10.2.1` | editable 构建需要 |
| CUDA 运行库 | `nvidia-cublas-cu12 12.9.1.4` / `nvidia-cuda-runtime-cu12 12.9.79` / `nvidia-cuda-nvcc-cu12 12.9.86` / `nvidia-cudnn-cu12 9.17.1.4` / `nvidia-nccl-cu12 2.28.9` / `nvidia-nvshmem-cu12 3.4.5` / `nvidia-cutlass-dsl 4.6.0` | **全部 cu12 系列**,与 12.8 驱动匹配 |

**只有 `vllm-omni` 是 editable / 本地路径安装。** 其余全部是普通 wheel 安装到
`/workspace/h3/env/lib/python3.12/site-packages`。这一条直接决定复现方式:
第 4 节的两个补丁都是改 `src-vllm-omni` 里的源码,**改完不需要重装**,重启引擎即生效。

完整清单已落盘:
`doc/machines/locks/runpods-5090x4.env.pip-freeze.txt`(225 条 + 7 行采集抬头)。
文件里唯二的 URL 是 vllm 的公开 release 直链和 vllm-omni 的公开 git 地址,**不含任何 token**
(已用 `scripts/check_repo_clean.sh` 同款正则扫过,0 命中)。本机无 conda,故没有 `.conda-env.yml`。

---

## 4. 第三方源码与补丁

### 4.1 checkout 清单(全机只有两个 git 工作区)

| 路径 | remote | branch | HEAD | dirty |
|---|---|---|---|---|
| `/workspace/h3/src-vllm-omni` | `https://github.com/vllm-project/vllm-omni.git` | **detached HEAD**(本地另有分支 `pr5910` 指向同一 commit,`main` 在 `593b4045`) | **`1a9b9c2c13c97763567405f43c5fee994c43ab13`**(短 `1a9b9c2c`;`git describe` = `v0.26.0-58-g1a9b9c2c`;commit 日期 2026-08-08 18:37:23 +0000,作者 lishunyang12,标题 `docs: add MiniMax-H3 BF16 and FP8 outputs`) | **1 行** `M vllm_omni/diffusion/offloader/distributed_layerwise_backend.py` |
| `/workspace/h3`(项目根本身) | `https://github.com/stargazerscripter123/H3_inference.git` | `main` | `39c4b5089f78f25ae62b5df19fa2e5f9492768c8` | 0(干净) |

`1a9b9c2c` 是上游 **PR #5910** 的 head,通过 `git fetch origin pull/5910/head:pr5910` 拉下来的。

- **没有 SGLang checkout**(全机搜不到)。
- **没有 ComfyUI,没有 custom_nodes**(`/workspace/h3` 下无 `ComfyUI/`;`comfy-kitchen` / `comfy-aimdo` / TeaCache / Turbo-LoRA 节点一律不存在)。因此"custom_nodes 逐个列出"在本机为**空集**。
- 两个 `.git/config` 的 remote 都是**匿名 https 公开地址**,没有嵌 token,也没有配 credential helper(`/workspace/h3/.git/credentials` 与 `~/.git-credentials` 均不存在)。

> 与历史档案的冲突 ②(重要):5090 上 stride 补丁被**固化成了分支** `pr5910-stride-fix @ 070096bd`,
> 而**本机是未提交的工作区修改**(`git status --short` 恰好 1 行 M)。
> 后果:在 `src-vllm-omni` 里执行 `git checkout -- .` / `git reset --hard` / `git stash` /
> 切分支,都会**静默把补丁抹掉**,而引擎不会报错、只会开始产噪声。
> 恢复办法是重新 `git apply /workspace/h3/scripts/pr5910_resident_stride_fix.patch`。

### 4.2 补丁表

| # | 补丁 | 改什么 | 为什么 | 不打的后果 | 判定命令 | **本机当前状态** |
|---|---|---|---|---|---|---|
| 1 | `scripts/pr5910_resident_stride_fix.patch`(md5 `56140c4d42d5e2f26cbd35386d401fd8`,1412 B) | `vllm_omni/diffusion/offloader/distributed_layerwise_backend.py` 的 `class PinnedResidentLayerGroup.load()`,hunk `@@ -620,9 +620,18 @@`:把 `gpu_buffer[off:off+numel].view(meta["shape"])` 换成 `torch.as_strided(..., size=meta["shape"], stride=meta["stride"])` | `_shard_and_pin` 按**物理(列主序)顺序**打包非连续权重(在线 FP8 的转置权重)并保存 stride;streamed 路径 `prefetch_layer` 用 `as_strided` 还原是对的,**常驻组却用 `.view()` 按行主序重解释** → 常驻 block 的所有 linear 权重值级乱序 | **不报错,静默产出纯噪声**(宽松 kernel)或直接崩(严格 kernel)。本机 FP8 路线 resident=50 = **全部 50 层常驻**,所以 100% 命中 | `sed -n '/class PinnedResidentLayerGroup/,/def offload/p' <文件> \| grep -q as_strided` | ✅ **已打**(实测:类内 `grep -c as_strided` = 2,`git apply --check --reverse` 成功)。日志抬头虽写着 `stride_patch=APPLIED_as_strided`,但**那个字段恒为 APPLIED,不算证据**,见下方警告 |
| 2 | `_ASYNC_OUTPUT_TIMEOUT` 上调 | `vllm_omni/diffusion/diffusion_engine.py:58` 的 `_ASYNC_OUTPUT_TIMEOUT = 30.0`(用在 :326 `step_streaming` 与 :793) | 这是"等引擎吐下一个输出"的硬上限,与我们设的 `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800` **是两回事,后者管不到它** | 超时抛 `TimeoutError` → HTTP 500,body 是**一句没内容的** `{"error":{"message":"Video generation failed:"}}`。实测本机 BF16 基座 NFE11(~30s 出头)两种拓扑都挂 | `grep -n _ASYNC_OUTPUT_TIMEOUT <文件>` | ❌ **未打**,当前仍是 `30.0`。与档案一致(三台机器源码里都是 30.0,不是本机配错);档案里"改成 600.0"是**建议**,本机没执行 |

> ⚠️ **补丁状态必须按代码内容判定,绝不能用 `git diff`。** 在 5090 上补丁已 commit,worktree 干净,
> `git diff --quiet` 会反过来报"没打"。
>
> ⚠️⚠️ **但日志抬头的 `stride_patch=` 字段不可信 —— 它用的就是本节下面痛批的那个裸 grep。**
> `h3_switch_runpods.sh:105-106` 是:
>
> ```bash
> STRIDE="MISSING_stride_patch"
> grep -q "as_strided" "$DLO" 2>/dev/null && STRIDE="APPLIED_as_strided"
> ```
>
> 与 `setup_runpods.sh:43` 是**同一个裸 grep**。未打补丁的原始文件本来就在 299/412 行各有一处
> `as_strided`(实测 `git show HEAD:… | grep -n as_strided` → 299、412;类内 `grep -c` → 0),
> 所以这个条件**恒真、无鉴别力**:`MISSING_stride_patch` 只有在 `$SRC` 或该文件整个不存在时才可能出现。
> 补丁被 `git checkout/reset` 抹掉后,日志照样写 `APPLIED_as_strided`。
> **在 `h3_switch_runpods.sh:106` 改成类内判定之前(见第 8 节待修缺陷),每次动过 `src-vllm-omni`
> 都必须手工跑一次类内判定:**
>
> ```bash
> DLO=/workspace/h3/src-vllm-omni/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py
> sed -n '/class PinnedResidentLayerGroup/,/def offload/p' "$DLO" | grep -c as_strided   # 期望 >=1;0 = 补丁没了
> ```

#### 关于 `setup_runpods.sh` 的探测缺陷 —— 本次给出了实证

`scripts/setup_runpods.sh:43` 的探测是**坏的**:

```bash
grep -q "as_strided" vllm_omni/diffusion/offloader/distributed_layerwise_backend.py \
  || echo "WARNING: check resident-group repoint for the stride bug before serving"
```

它只在整个文件里找 `as_strided`。实测证明这个条件**恒真、因此补丁永远不会被打**:

```
# 未打补丁的原始文件(git show HEAD:...)
PRISTINE_as_strided_count=2          ← 第 299 行 physical_view、第 412 行 prefetch_layer
PRISTINE_class_scoped=0              ← PinnedResidentLayerGroup 类内一处都没有
```

即:**文件别处本来就有两处 `as_strided`**,裸 grep 必中,`||` 分支永不执行,连那句 WARNING 都不会打印。
`post_setup.sh:19-20` 与 `post_setup2.sh:13-15` 的探测才是对的(用
`sed -n '/class PinnedResidentLayerGroup/,/def offload/p' | grep -q as_strided` 限定在类内),
本机当前这份补丁就是 `post_setup2.sh` 打上的(`logs/post_setup2.log` 有记录)。

---

## 5. 模型权重

磁盘口径要拆成三个数,别混着用(实测 `du -sh`):

- **真权重稳态 197G** = base `FL2VA` 135G + `merged` 62G。这是长期必须留的。
- **下载过程峰值约 253G** = 197G + 下载残留 56G(base `.cache` 34G + loras `.cache` 22G)。
  规划新 pod 的盘按这个数留余量。
- **整个 `/workspace/h3` 现为 263G** = 253G + venv 12G(± 日志/输出零头);
  `df -h /workspace` = 503G 用 263G 余 215G(56%)。

清掉那 56G 残留后即回到 209G 稳态(`/workspace` 从 56% 降到约 45%)。

| 路径 | 体积 | 来源(HF repo @ revision) | 用途 | 校验契约 |
|---|---|---|---|---|
| `/workspace/h3/base/MiniMax-H3/FL2VA` | **135G** | `MiniMaxAI/MiniMax-H3`,`hf download --include "FL2VA/*"`;**本机 HF 缓存 `refs/main` = `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`** | 4 个 `*-original-*` profile 的模型根 | `post_setup.sh` 硬校验 `transformer/*.safetensors` 数量 == 13;`transformer/model.safetensors.index.json` sha256 = `fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb` |
| ├ `transformer/` | 62G | 13 shard(`model-000{01..13}-of-00013.safetensors`)+ index | DiT,50 层 | 同上 |
| ├ `text_encoder/` | 63G | 14 shard + index + tokenizer 文件 | Qwen3-VL TE(BF16) | —— |
| ├ `video_vae/` | 9.8G | `source/model.safetensors` + 一组自定义 `.py`(klvae/vae_vit/parallel…) | video VAE | —— |
| ├ `audio_vae/` | 578M | 1 个 safetensors | audio VAE | —— |
| └ `processor/` `tokenizer/` | 各 11M | —— | —— | —— |
| `/workspace/h3/merged/MiniMax-H3-Turbo-v4s600ema` | **62G** | **本机 CPU 上重做的 merge**(`post_setup2.sh` → `merge_turbo_lora.py`)。`FL2VA/transformer` 是真实文件,`audio_vae`/`processor`/`text_encoder`/`tokenizer`/`video_vae` 都是指回 base 的 symlink | 4 个 `*-turbo-*` profile 的模型根 | 根目录有 **`.complete`**(内容 `2026-08-09T01:45:43+0000`)。`h3_switch_runpods.sh` 启动 turbo 前检查 `[ -f .../.complete ]`,缺则拒绝启动。另有 `merge_manifest.json`、`delta_norms.csv` |
| `/workspace/h3/loras/MiniMax-H3-Turbo-Lora/minimax_h3_turbo_v4_step600_ema.safetensors` | 779 849 816 B(≈780M) | `larryvrh/MiniMax-H3-Turbo-Lora` @ **`afc0346516372a17162c14df3c5264de1d9aa1c0`**(HF 缓存 `refs/afc0346` 指向它) | merge 的输入 | **sha256 = `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`**(本次实测全值;历史档案只记到前 8 位) |
| `/workspace/h3/base/MiniMax-H3/.cache` | **34G** | HF/xet 下载缓存残留 | 无用 | **可回收** |
| `/workspace/h3/loras/MiniMax-H3-Turbo-Lora/.cache` | **22G** | 同上,`.../download/` 下 8 个 `*.incomplete` 大 blob(2.4–3.1G 不等) | 无用 | **可回收** |

> 与历史档案的冲突 ③:档案写 "base/MiniMax-H3/FL2VA 官方 BF16 **168G**"。
> 实测 **`FL2VA` 本体只有 135G**,`du -sh base/` 的 168G 里含 34G 缓存残留。168G 这个数
> 正是 `post_setup.sh` 打进日志的 `du -sBG $B/base`,所以档案抄的是"含缓存"的口径。
>
> 与历史档案的冲突 ④:档案说 "LoRA 仓整仓拉 22 文件 = 白下 22G",听上去像已经解决了。
> **实测那 22G 现在还躺在盘上** —— 第一次整仓下载的 `.incomplete` blob 从没被清理。
> 加上 base 的 34G,**共 56G 可以直接删,`/workspace` 立刻从 56% 降到约 45%**。
> (删除属于写操作,本次盘点只读,未执行。)

### merge_manifest.json 关键字段(本机实测全文)

```json
{
  "base_model": "MiniMaxAI/MiniMax-H3",
  "base_path": "/workspace/h3/base/MiniMax-H3/FL2VA",
  "base_transformer_index_sha256": "fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb",
  "lora_repo": "larryvrh/MiniMax-H3-Turbo-Lora",
  "lora_revision": "afc0346",
  "lora_file": "minimax_h3_turbo_v4_step600_ema.safetensors",
  "lora_sha256": "5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3",
  "strength": 1.0,
  "alpha_convention": "absent -> alpha=rank (scale 1.0)",
  "merge_dtype": "float32",
  "output_dtype": "bfloat16",
  "qkv_disk_layout":    "group-interleaved (56 groups x [q|k|v] x 128)",
  "qkv_runtime_layout": "Q_all_K_all_V_all (LoRA lora_B order)",
  "merge_script_sha256": "b034c41b1c20233333890d5c168583de48730fa9f7df10c2c4041c615b5dcb1b",
  "modified_tensor_count": 259,
  "total_tensor_count": 535,
  "created": "2026-08-09T01:45:43+0000",
  "verification": "L1 bit-level recompute; L2 Comfy single-file layout oracle; L3 single-layer forward oracle (cos>0.9999, relL2<5e-3)"
}
```

`logs/post_setup2.log` 里的实际校验结论:

```
all 259/259 LoRA targets applied
verify L1 passed: 259 modified + 276 unmodified = 535 tensors bit-checked
verify L2 SKIPPED (no --comfy-single) — layout not independently confirmed!
verify L3 passed: worst min_row_cos=0.999999, worst relL2=9.65e-04
top-10 |dW|_F/|W|_F: blocks.49.mlp.fc2.weight=0.0036; blocks.49.attn.qkv_proj.weight=0.0023; …
```

L2(Comfy 单文件独立布局 oracle)在本机**没有跑**,因为这台机器没有 ComfyUI 单文件权重
(调用时传的是 `--comfy-single /nonexistent`)。布局正确性靠**与 6000a 的溯源哈希比对**替代。

### 三机 merge 溯源交叉验证(本次实跑,顺带解掉一个档案坑)

| | 6000a | 5090-Runpods |
|---|---|---|
| HF 缓存 `refs/main` | `b3c7290e66afdf293bef3b9077b7a266ef421f34` | **`bfc8ed0353f5a9733be73e6b2c98ec0948195b86`** |
| `base_transformer_index_sha256` | `fb457a26ffa629…a29a7efb` | **`fb457a26ffa629…a29a7efb`(相同)** |
| `lora_sha256` | `5f3a626cd72c93…07a416d3` | **相同** |
| `merge_script_sha256` | `b034c41b1c2023…15b5dcb1b` | **相同** ——(⚠️ 但**不可复现**,见下方专门说明,不要拿它当门禁) |
| `modified/total` | 259 / 535 | **259 / 535** |
| `lora_revision` 记法 | 全长 `afc0346516372a17162c14df3c5264de1d9aa1c0` | 短名 `afc0346`(同一 commit,只是记法不同) |
| **merged transformer 分片 sha256** | **未采集** | **未采集** ——(⚠️ 缺的正是唯一能证明位级等价的那一项) |

> 与历史档案的冲突 ⑤(**重要,会误导后来人**):档案的"上游 pin 速查表"里写着
> **"官方权重 snapshot `MiniMaxAI/MiniMax-H3` sha = `b3c7290e…`"**,像是一个可以钉死的 pin。
> **实测本机是 `bfc8ed03…`**,而采集当天 10:05 上游又走到了 `6818f6c3…` —— 一天之内动了两次。
> 但**两边的 `FL2VA/transformer` index sha256 完全相同**。
>
> 结论:**snapshot sha ≠ 权重身份**。档案里那条 pin 应当降级为"某次下载的记录"。
>
> ⚠️ **但 index sha256 相同也只是强提示,不构成"位级等价"的证明。**
> 实测该文件的全部内容只有两个 key:`metadata`(仅 `{"total_size": 66280430144}`)和
> `weight_map`(535 条 `张量名 → model-000NN-of-00013.safetensors` 映射)——
> **里面没有任何分片摘要**。所以 index sha256 相同,严格只能推出
> **张量名 / 分片划分 / 总字节完全一致**;上游任何一次"保持张量名、形状、分片、总字节不变"的
> 重传(重新量化、微调后重传、只改元数据)都会产出**字节完全相同**的 index.json。
>
> **待补的真 pin(本次未采集)**:13 个 `transformer/model-000*-of-00013.safetensors` 的 sha256
> (或至少 merged 输出分片的 sha256),记进 `merge_manifest` 与本表。
> 在补上之前,"三机权重位级等价、可互引 benchmark"这个结论**只是高置信推测,不是已证事实**
> (推测,依据:index.json 的张量名/分片/总字节三机一致 + merge manifest 其余字段一致 +
> merge 自校验 L1 259/535 bit-checked)。
>
> ⚠️ **`merge_script_sha256` 三机相同,但这一项是"死"的、不可复现。**
> 该字段是**脚本自身文件的 sha256**(`merge_turbo_lora.py:349` 算、`:367` 写)。
> manifest 里的 `b034c41b…` 对应的那份脚本**已不存在于任何机器,也从未进过 git 历史**:
> 全机 `find` 只有一份 `merge_turbo_lora.py`,三处(runpods / 6000a / Mac)哈希一致
> = `70cdfe9d7b1e1d…e60ba777d`,且其 mtime(08-09 08:25)**晚于**合并发生的时间(01:36–01:45),
> 即合并之后脚本被覆盖过;`git log -- scripts/merge_turbo_lora.py` 只有一次"初始导入 `d66049a`"。
> 所以**用仓库现版本重跑,该字段必然是 `70cdfe9d…`,与两机 manifest 对不上,这是预期的**,
> 不代表合并出错。见第 8 节待修缺陷。

### HF 缓存布局

```
HF_HOME = /root/hf-cache        (24M,只有 refs,没有 blobs —— 走的是 --local-dir 直下)
  hub/models--MiniMaxAI--MiniMax-H3/refs/main            -> bfc8ed03…
  hub/models--larryvrh--MiniMax-H3-Turbo-Lora/refs/afc0346 -> afc0346516372a17162c14df3c5264de1d9aa1c0
  xet/  24M
  token                     ← 37 字节。**token 值不入库**;来源见 credentials/HF.md
/root/.cache/huggingface/token  ← 同样 37 字节的副本
```

> ⚠️ 两个 token 文件都在 `/`(临时盘)上,**pod 重建即失**。这是设计如此:
> 重建时由 `setup_runpods.sh <HF_TOKEN>` 重新写入。
> 本机**没有** 6000a/5090 那种"HF cache 桥接 symlink"(那是为 SGLang 做的,本机没装 SGLang)。

---

## 6. 运行期环境变量

全部来自 `scripts/h3_switch_runpods.sh` 的 `_launch`(第 113-117 行的 `setsid nohup env …`)。
本机 `/root/.bashrc` 里**没有**任何 H3 相关的 export,登录 shell 环境里也没有
`HF_*` / `CUDA_*` / `VLLM_*` —— 一切都在启动那一行里显式给,这是好事(可复现)。

| 变量 | 值 | 作用 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 四卡全给引擎。TP2 时 vLLM 内部把 rank 0-1 / 2-3 配成一组,正好落在同 NUMA 的 (0,1)(2,3) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | **本机的常规 switch 脚本里默认设**;两台本地机只在 `tp2_r2_*` 一类实验脚本里设(`tp2_r2_6000a.sh:23` / `tp2_r2_launch_p2.sh:11` / `tp2_r2_launch_tp1.sh:10`),常规 switch 没设。DLO 反复搬运常驻块会造成显存碎片,可扩展段能显著降低碎片导致的假 OOM |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | 多进程 worker 用 spawn 而非 fork,避免 CUDA context 被 fork 破坏 |
| `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` | `1800` | 视频合成阶段的同步等待上限(秒)。**注意它管不到引擎内部那个 30 秒硬超时**(见 4.2 补丁 2) |
| `VLLM_DISABLED_KERNELS` | `${VLLM_KERNELS_DISABLE:-CutlassFP8ScaledMMLinearKernel}` | **sm_120 上禁 Cutlass FP8 scaled-mm**。5090 上这条 kernel 会崩(`scaled_mm_entry.cu:209`),禁掉后回落到 ChannelWiseTorch |
| `VLLM_ATTN`(脚本变量) | 默认 `CUDNN_ATTN` | 传给 `--diffusion-attention-backend`。日志实证:`Resolved diffusion attention backend 'CUDNN_ATTN'`。**别换 TORCH_SDPA**(5090 上实测慢约 18s) |
| `H3_TE_TP`(脚本变量) | 默认 `4` | 传给 `--text-encoder-tp-size`。**tp2u2 时必须 = 4(世界大小)**,传 2 会在 `_build_text_encoder_group` 的 `assert cpu_group is not None` 处全崩 |
| `VLLM_KERNELS_DISABLE` | 未设(用默认) | 覆盖上面那条禁用列表的逃生口 |

完整启动命令(实测运行中的进程):

```
vllm serve /workspace/h3/merged/MiniMax-H3-Turbo-v4s600ema/FL2VA \
  --omni --host 127.0.0.1 --port 8091 --trust-remote-code \
  --num-gpus 4 --tensor-parallel-size 4 --text-encoder-tp-size 4 --usp 1 --ring 1 \
  --quantization fp8 \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 50 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN --enforce-eager
```

`resident` 默认值由精度决定:**bf16 → 40,fp8 → 50**(`h3_switch_runpods.sh` 的 `DEFRES`)。
H3 的 DiT 恰好 50 层(`transformer/config.json` 的 `"num_layers": 50`),所以 fp8 的 r50 =
**全部常驻**,日志印证:`Keeping 50 leading blocks resident on transformer; streaming 0 tail blocks` /
`All blocks for transformer are resident; no streaming hooks required`。

> ⚠️ 正因为"全常驻",4.2 的 stride 补丁在本机是 **100% 命中**的路径,不是边角情况。

`run/` 下的状态文件(运维铁律的落地):`vllm.pid` / `vllm.variant`(如 `turbo-fp8-tp4-r50`)/
`vllm.model`(模型绝对路径)/ `switch.lock`。`:8091` 被所有变体复用,`vllm.variant` 是
防"静默服错 checkpoint"的唯一凭据。

---

## 7. 从零复现(pod 重建后照着敲)

前提:新 pod 已挂好持久卷到 `/workspace`,能 ssh 进去。

> **关于 HF token:两个 repo(`MiniMaxAI/MiniMax-H3`、`larryvrh/MiniMax-H3-Turbo-Lora`)
> 实测均为公开、非门控**(`curl -s https://huggingface.co/api/models/<repo>` → `private:False, gated:False`),
> 匿名下载即可,**token 非必需**,只影响限速/配额。有就按步骤 1 写进 `/root/hf-cache/token`
> (值见 `credentials/HF.md`,不入库);没有可以整段跳过,不要为此卡在起步。

### 步骤 0 — 先验驱动,再决定装哪套轮子

```bash
nvidia-smi                       # 看 Driver Version 与 CUDA Version
nvidia-smi topo -p2p r           # 看 P2P;全 CNS 则 TP 分组必须同 NUMA 配对
lscpu | grep -i numa             # 确认哪几张卡在哪个 NUMA
free -g                          # 期望 ~1007G;显著更小就别指望 DLO 常驻方案
df -h /workspace                 # 期望 ~503G;至少要留 280G 余量(见下)
which ffmpeg                     # **硬依赖**,本 pod 镜像自带 6.1.1;没有则 apt-get install -y ffmpeg
```

判据:**驱动 <580 → 用 cu129 轮子(下面这套);驱动 ≥580 → 直接 `pip install vllm==0.26.0`**。

磁盘判据按**峰值**而不是稳态算:真权重稳态 197G,但下载过程中 xet 残留会额外占 56G,
再加 venv 12G,**至少留 280G 余量**(拆解见第 5 节开头)。

> ⚠️ 别跳过 `topo -p2p`。这台机器"没有 P2P"是所有拓扑结论(tp2u2 快过 tp4)的前提,
> 换一台有 NVLink 的机器结论会反过来。
>
> ⚠️ `ffmpeg` 是**系统级**硬依赖(`/usr/bin/ffmpeg`,apt 装的,不是 venv 里的 imageio-ffmpeg):
> `probe.sh:13` 与 `bench_matrix.sh:39` 都直接调它抽质量帧。而 `probe.sh` 用 `&&` 串联 ——
> **缺 ffmpeg 时不报错,只是不打印 `FRAME_OK`**。看不到 FRAME_OK 先查 ffmpeg,别去查引擎。

### 步骤 0.5 — 先把项目仓库拉下来(**必须在步骤 1 之前**)

步骤 4 的补丁文件、步骤 7 的 merge 脚本、步骤 8 的 switcher/probe/bench 全都来自这个仓库,
所以它必须**第一个**到位。`/workspace/h3` 本身就是 git 工作区:

```bash
mkdir -p /workspace/h3 && cd /workspace/h3
git init
git remote add origin https://github.com/stargazerscripter123/H3_inference.git
git fetch --depth=1 origin main
git checkout -f -B main origin/main      # 当前 main @ 39c4b508
```

> ⚠️ **为什么不用 `git clone`**:步骤 1 会 `mkdir -p` 一堆子目录,`git clone` 到非空目录会失败;
> 而全新 pod 上 `/workspace` 是空的、没有 `.git`,`git pull` 也无从谈起。上面这套
> `init + remote add + fetch + checkout -f` 对**空目录和非空目录都成立**,是唯一稳妥的写法。
>
> ⚠️ **该 repo 公开,匿名 https 即可,不需要任何凭据**(实测
> `GIT_TERMINAL_PROMPT=0 git ls-remote --heads <url>` rc=0;本机既无 `~/.git-credentials`
> 也没配 credential helper —— 这个状态要保持)。
>
> ⚠️ `setup_runpods.sh` 抬头注释说这些脚本"是从 Mac 推过去的",那是**早期的引导方式**;
> 现在项目根就是这个 git 工作区,以本步骤为准。

### 步骤 1 — 目录与环境

```bash
B=/workspace/h3
mkdir -p $B/{base,merged,loras,outputs,logs,inputs,workflows,run} /root/hf-cache
#                                        ↑ 注意没有 scripts:它由步骤 0.5 的仓库提供,别自己建空的
printf '%s' "<HF_TOKEN>" > /root/hf-cache/token      # 可选;token 值来自 credentials/HF.md,不入库

python3 -m venv $B/env          # 系统 python 3.12.3;本机没有也不需要 conda
$B/env/bin/pip install -q --upgrade pip
$B/env/bin/pip install -q huggingface_hub        # **不要装 [hf_transfer] extra**,见步骤 6
```

> ⚠️ **不要照抄 `setup_runpods.sh:26` 的 `"huggingface_hub[hf_transfer]"`。** hf_transfer 已被
> 弃用(步骤 6 详述),而且现环境是 `fix_cu129.sh` 重建的、它 `:35` 装的就是不带 extra 的
> `huggingface_hub` —— 实测 freeze 与 lock 文件里**都没有 hf_transfer**,只有
> `huggingface_hub==1.27.0` + `hf-xet==1.6.0`。装了 extra 会与配套 lock 文件对不上。
>
> ⚠️ **venv 不可重定位**。`fix_cu129.sh` 走的是"建 `env129` → 验证 → 把旧 env 改名 →
> `mv env129 env`"的路子,但 venv 里 `bin/*` 的 shebang 和 `pyvenv.cfg` 都写死了创建时的绝对路径,
> `mv` 之后会报 `No such file or directory`(说的是**解释器**不是脚本)。
> 本机现存 env 的 `pyvenv.cfg` 写的是 `command = /usr/bin/python3 -m venv /workspace/h3/env`、
> shebang 是 `#!/workspace/h3/env/bin/python3`,即**最终是原地建的**(mtime 06:05,
> 晚于 `fix_cu129.sh` 完成的 05:44)。
> **建议:一开始就在最终路径 `/workspace/h3/env` 上建,不要建了再搬。**

### 步骤 2 — 装 torch + vLLM(版本必须钉死,理由见下)

```bash
WHL=https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
$B/env/bin/pip install "$WHL" --extra-index-url https://download.pytorch.org/whl/cu129
$B/env/bin/pip install setuptools_scm wheel aenum safetensors
```

为什么是这个变体、这个版本:

- **`+cu129` 而不是 PyPI 默认**:PyPI 的 `vllm==0.26.0` 会拉 `torch 2.11.0+cu130`,
  CUDA 13 需要 driver ≥580,本机 570 → 启动即 `driver too old (found version 12080)`。
  **`+cu128` 轮子不存在,别去找。**
- **`0.26.0` 而不是更新**:必须与 vllm-omni 0.26.0 配对。
- `--extra-index-url .../cu129` 是为了让 torch/torchvision/torchaudio 也解析到 cu129 变体
  (最终应得到 `torch 2.11.0+cu129` / `torchvision 0.26.0+cu129` / `torchaudio 2.11.0+cu129`)。

立即验证(不通过就别往下走):

```bash
$B/env/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_arch_list())"
# 期望: 2.11.0+cu129 12.9 True 4
#       [... 'sm_120', 'compute_120']     ← sm_120 必须在,否则 5090 跑不了
```

### 步骤 3 — vllm-omni 源码 checkout 到指定 commit

```bash
cd $B && [ -d src-vllm-omni ] || git clone https://github.com/vllm-project/vllm-omni.git src-vllm-omni
cd $B/src-vllm-omni
git fetch origin pull/5910/head:pr5910
git checkout 1a9b9c2c13c97763567405f43c5fee994c43ab13
```

为什么钉这个 commit:PR #5910(分布式 layer-wise offload,DLO)**尚未合并**,行为在不同 head 之间变过
(项目里出现过三个 head:`9f92147640…` 计划稿 / `b18eeff22a…` 5090+6000a 实际 / `1a9b9c2c13…` 本机)。
不钉死就无法解释 benchmark。**本机用 `1a9b9c2c`。** 注意这会让仓库处于 detached HEAD 状态,正常。

### 步骤 4 — 打 stride 补丁(**先判定,再打**)

```bash
cd $B/src-vllm-omni
DLO=vllm_omni/diffusion/offloader/distributed_layerwise_backend.py
if sed -n '/class PinnedResidentLayerGroup/,/def offload/p' "$DLO" | grep -q as_strided; then
  echo "已在上游修好,跳过"
else
  git apply $B/scripts/pr5910_resident_stride_fix.patch && echo "stride patch applied"
fi
# 复核
sed -n '/class PinnedResidentLayerGroup/,/def offload/p' "$DLO" | grep -c as_strided   # 期望 >=1
```

> ⚠️ **判定必须限定在 `PinnedResidentLayerGroup` 类内。** 裸 `grep as_strided <文件>` 恒为真 ——
> 未打补丁的原始文件在第 299 行(`physical_view`)和第 412 行(`prefetch_layer`)本来就各有一处。
> `scripts/setup_runpods.sh:43` 犯的就是这个错,**它从来没有打过补丁,连 WARNING 都不会打印**。
> 用 `post_setup2.sh` 里那段(限定类内)才是对的。
>
> ⚠️ **绝不能用 `git diff` 判定。** 在别的机器上补丁被 commit 过,worktree 干净时 `git diff` 会误报"没打"。
>
> ⚠️ 本机补丁是**未提交的工作区修改**。`git checkout -- .` / `reset --hard` / 换分支都会静默抹掉它,
> 而**引擎不会报错,只会开始产纯噪声**。要么每次改完复核一次,要么学 5090 把它 commit 成分支。

### 步骤 5 — editable 安装 vllm-omni

```bash
cd $B/src-vllm-omni
SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0 $B/env/bin/pip install -e . --no-build-isolation
$B/env/bin/python -c "import vllm, vllm_omni; print(vllm.__version__, vllm_omni.__file__)"
# 期望: 0.26.0 /workspace/h3/src-vllm-omni/vllm_omni/__init__.py
```

为什么这两个 flag:

- `SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0` —— 版本号由 setuptools_scm 从 git 推导;
  detached HEAD / 缺 `.git` 都会退化成 `dev`,构建失败。写死避免这一类。
  (历史上 rsync 源码时 `--exclude .git` 就踩过这个坑。)
- `--no-build-isolation` —— 构建脚本会 import 自身,需要依赖在场;隔离环境里没有。

**装成 editable 是有意为之**:第 4 节两个补丁都是改这份源码,editable 意味着**改完重启引擎即生效,
不用重装**。装成普通包会让补丁流程完全失效。

### 步骤 6 — 下权重

```bash
export HF_HOME=/root/hf-cache
export HF_XET_HIGH_PERFORMANCE=1          # 见下面的 ⚠️
$B/env/bin/hf download MiniMaxAI/MiniMax-H3 --revision bfc8ed0353f5a9733be73e6b2c98ec0948195b86 \
  --include "FL2VA/*" --local-dir $B/base/MiniMax-H3 &
$B/env/bin/hf download larryvrh/MiniMax-H3-Turbo-Lora --revision afc0346 \
  --include "minimax_h3_turbo_v4_step600_ema.safetensors" \
  --local-dir $B/loras/MiniMax-H3-Turbo-Lora &
wait
# 校验
ls $B/base/MiniMax-H3/FL2VA/transformer/*.safetensors | wc -l      # 必须 == 13
sha256sum $B/base/MiniMax-H3/FL2VA/transformer/model.safetensors.index.json
#   期望 fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb
sha256sum $B/loras/MiniMax-H3-Turbo-Lora/minimax_h3_turbo_v4_step600_ema.safetensors
#   期望 5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3
```

> ⚠️ **base 的 `--revision` 是"本机那次下载的 snapshot",不是权威 pin。** 上游 `main` 在动:
> 本机下的是 `bfc8ed03…`、6000a 下的是 `b3c7290e…`、而**采集当天 10:05 上游又前进到了
> `6818f6c3…`**(实测 `curl -s https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3` 的 `sha`)。
> 不带 `--revision` 直接下,现在大概率与下面的 index sha256 对不上。
> **真正的判据是 index sha256(+ 分片 sha256,见步骤 7)**,revision 只是让你能重现同一次下载。
>
> **恢复路径(index sha256 对不上时):** ① 先用上面写死的 `--revision bfc8ed03…` 重下;
> ② 仍不符 → 说明连那个 snapshot 都取不到或子树确实变了,**停下来重新做一次溯源**
> (重新记录 index sha256 + 分片 sha256,并在文档里更新),
> **绝对不要在这种状态下直接引用别机的 benchmark**。
>
> ⚠️ **`HF_HUB_ENABLE_HF_TRANSFER=1` 已被弃用,用它会走退化路径(~28MB/s)。**
> 本机 `logs/dl_base.log` 第一行就是 huggingface_hub 打的
> `FutureWarning: ... deprecated as 'hf_transfer' is not used anymore. Please use HF_XET_HIGH_PERFORMANCE instead`。
> **`scripts/setup_runpods.sh` 里至今还用的是旧的那个,这是已知待修缺陷。**
> 换成 `HF_XET_HIGH_PERFORMANCE=1` 峰值可到 247MB/s。
> **排期按端到端约 2h15m 算**(本机 22:47 起、`logs/post_setup.log` 记 `[01:01:11] base download done`):
> 第一趟走旧 hf_transfer 退化路径 ~28MB/s,第二趟**断点续传**才是 `dl_base2.log` 里那句
> `Fetching 81 files: 100%|…| 81/81 [1:24:16<00:00]` —— **1h24m 只是续传那一趟,不是端到端**
> (同一行开头就瞬间到 30%,即大部分文件上一趟已经落盘)。全程用 HF_XET_HIGH_PERFORMANCE 会明显更快。
>
> ⚠️ **LoRA 必须带 `--include`。** 那个仓有 22 个实验 checkpoint,整仓拉 = 白下 22G。
> 本机就是这么踩的,而且那 22G 的 `.incomplete` blob **至今还在
> `loras/MiniMax-H3-Turbo-Lora/.cache/huggingface/download/` 里**。下完记得清:
> `rm -rf $B/base/MiniMax-H3/.cache $B/loras/MiniMax-H3-Turbo-Lora/.cache`(共约 56G)。

### 步骤 7 — 合并 Turbo LoRA

```bash
$B/env/bin/python $B/scripts/merge_turbo_lora.py \
  --lora $B/loras/MiniMax-H3-Turbo-Lora/minimax_h3_turbo_v4_step600_ema.safetensors \
  --base $B/base/MiniMax-H3/FL2VA \
  --dst  $B/merged/MiniMax-H3-Turbo-v4s600ema \
  --comfy-single /nonexistent --lora-revision afc0346
```

耗时约 9 分钟(本机 01:36→01:45)。为什么走 merged checkpoint 而不是运行期 LoRA:
vLLM-Omni 对 fused 层(qkv / fc1)的 runtime LoRA 会 **warning 后跳过**,蒸馏静默失效。
`--comfy-single /nonexistent` 是**有意跳过 L2 校验**(本机没有 ComfyUI 单文件权重),
布局正确性靠与 6000a 的 manifest 哈希比对替代。

预期输出(与本机 `logs/post_setup2.log` 一致):

```
all 259/259 LoRA targets applied
verify L1 passed: 259 modified + 276 unmodified = 535 tensors bit-checked
verify L2 SKIPPED (no --comfy-single) — layout not independently confirmed!
verify L3 passed: worst min_row_cos=0.999999, worst relL2=9.65e-04
DONE -> /workspace/h3/merged/MiniMax-H3-Turbo-v4s600ema (manifest + .complete written)
```

**跑完必须核对 manifest。** 门禁分三档,别混为一谈:

| 项 | 是不是门禁 | 说明 |
|---|---|---|
| `base_transformer_index_sha256` | ✅ 必须相同 | 证明**张量名/分片划分/总字节**一致 |
| `lora_sha256` | ✅ 必须相同 | LoRA 输入文件位级相同 |
| `modified/total` = `259 / 535` | ✅ 必须相同 | 命中的 LoRA target 数一致 |
| **merged transformer 分片 sha256** | ✅ **应当相同(目前尚未采集,见下)** | **这才是"位级等价"的唯一证据** |
| `merge_script_sha256` | ❌ **不是门禁,必然不同** | 见下面的说明,对不上是**预期行为** |

> ⚠️ **`merge_script_sha256` 对新人永远对不上,这是已知情况,不是你合并出错。**
> 该字段的定义是**脚本自身文件的 sha256**(`scripts/merge_turbo_lora.py:349`
> `subprocess…(["sha256sum", os.path.abspath(__file__)])`,写进 manifest 在 `:367`)。
> 两机 manifest 记的都是 `b034c41b1c2023…15b5dcb1b`,但**那一版脚本已经不存在于任何机器、
> 也从未进过 git 历史**(`git log -- scripts/merge_turbo_lora.py` 只有一次"初始导入";
> 全机 `find` 只有一份文件;其 mtime 2026-08-09 08:25 晚于合并发生的 01:36–01:45,
> 即脚本在合并**之后**被覆盖了)。
> 仓库现版本三处一致(runpods / 6000a / Mac)= `70cdfe9d7b1e1d…e60ba777d`。
> **用现版本重跑,该字段必然是 `70cdfe9d…`,与两机 manifest 的 `b034c41b…` 不同 —— 忽略它。**
>
> ⚠️ **不要用 HF repo 的 snapshot sha 来判断"权重是不是同一份"。** 本机与 6000a 的
> `refs/main` 不同(`bfc8ed03…` vs `b3c7290e…`)但 transformer index sha256 相同。
>
> ⚠️ **也不要把 index sha256 相同当成"位级等价"。** 见第 5 节交叉验证表下的说明:
> `index.json` 里只有 `total_size` + `weight_map`,**没有任何分片摘要**。
> 要断言位级等价,必须再比分片 sha256:
>
> ```bash
> # 在两台机器上各跑一次,逐行对比(注意:这是 62G 全盘读,别在别人跑任务时做)
> cd $B/merged/MiniMax-H3-Turbo-v4s600ema/FL2VA/transformer && sha256sum model-000*-of-00013.safetensors
> ```

### 步骤 7.5 — 准备冒烟输入(**别跳过,跳了步骤 8 必失败**)

`probe.sh` 与 `bench_matrix.sh` 都**硬编码**了三个输入路径:

```
$B/inputs/first_864x480.png
$B/inputs/last_864x480.png
$B/workflows/smoke_prompt.txt
```

其中 `workflows/smoke_prompt.txt` 在 git 里(步骤 0.5 已经带下来了),
但**两张 864×480 PNG 不在 git 里** —— 实测 `git ls-files inputs` 只有 `inputs/.gitkeep`,
步骤 1 的 `mkdir -p $B/inputs` 只建了个空目录。**必须自己准备**:

```bash
# 方式 A(推荐):从两台本地机拷这次 benchmark 用的那两张,保证跨机可比。
# 注意 ~ 要用单引号包住,否则会被 Mac 本地 shell 展开成 /Users/ning。
scp popos-6000a:'~/data/dropbox/CV/h3/inputs/first_864x480.png' $B/inputs/
scp popos-6000a:'~/data/dropbox/CV/h3/inputs/last_864x480.png'  $B/inputs/
# popos-5090 上是同样两份(字节数一致),二选一即可

# 方式 B:自己造两张 864×480 的首/尾帧(此时 benchmark 只能自比,不能与本文档的数字对照)
```

前置校验(步骤 8 之前跑一遍,三个文件都要在):

```bash
ls -l $B/inputs/first_864x480.png $B/inputs/last_864x480.png $B/workflows/smoke_prompt.txt
# 三机一致的字节数:first_864x480.png = 529358,last_864x480.png = 585621
```

> ⚠️ 缺这两张图时的失败是 `FileNotFoundError`,**和引擎、驱动、补丁全都无关**,
> 但很容易被误判成环境问题。先看这一行。
>
> ⚠️ 换图会改变 benchmark 数字。要引用第 8 步那张表,输入必须是同两张图。

### 步骤 8 — 更新脚本 + 冒烟验证

脚本在步骤 0.5 就已经到位了。这里只是**在改动之后同步最新版**:

```bash
cd /workspace/h3 && git pull        # 当前在 main @ 39c4b508
```

启动 + 冒烟:

```bash
bash $B/scripts/h3_switch_runpods.sh turbo fp8 tp2u2 50
head -3 $B/logs/vllm_server.log
```

预期日志抬头(**这三行是最重要的自证**)。下面是本机 2026-08-09 的**真实日志原文**,
注意它是 `turbo fp8 tp4` 那次的记录,所以 `label`/`tp`/`usp` 显示的是 tp4;
按上面 `tp2u2` 参数启动时对应位置应为 `label=turbo-fp8-tp2u2-r50 … tp=2 usp=2`:

```
### h3_switch_runpods launch 2026-08-09T09:35:25+00:00
### model=/workspace/h3/merged/MiniMax-H3-Turbo-v4s600ema/FL2VA  label=turbo-fp8-tp4-r50  resident=50  tp=4 usp=1
### src=/workspace/h3/src-vllm-omni head=1a9b9c2c stride_patch=APPLIED_as_strided
```

> ⚠️⚠️ **不要拿抬头里的 `stride_patch=` 当门禁 —— 它恒为 `APPLIED_as_strided`。**
> `h3_switch_runpods.sh:106` 用的是裸 `grep -q as_strided <整个文件>`,而未打补丁的原始文件
> 本来就在 299/412 行各有一处,所以这个字段**永远不会报 `MISSING`**(除非 `$SRC`/文件整个不见了)。
> 详见 §4.2 的警告与第 8 节待修缺陷。
>
> **真正的门禁是类内判定,每次启动前手工跑一次:**
>
> ```bash
> DLO=$B/src-vllm-omni/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py
> sed -n '/class PinnedResidentLayerGroup/,/def offload/p' "$DLO" | grep -q as_strided \
>   && echo "STRIDE_OK" || echo "STRIDE_MISSING —— 立刻停,回步骤 4"
> ```
>
> 输出 `STRIDE_MISSING` 就**立刻停,回步骤 4**;抬头那三行只用来确认 `label` / `tp` / `usp` /
> `head` 服的是不是你想要的那个 checkpoint。

等健康 + 出片:

```bash
for i in $(seq 1 120); do curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:8091/health | grep -q 200 && break; sleep 10; done
bash $B/scripts/probe.sh smoke 0 6
```

预期:**冷启动约 175–180s 到 `/health` 200**(本机实测:抬头 `launch 09:35:25` →
`api_server.py:548 Starting vLLM API server` 09:38:18,紧随 `Application startup complete.`,
即 **173s**)。上面那个等待循环是 120×10s = 1200s,余量充足,不用改。
`probe.sh` 打印 `wall=…` 并输出 `outputs/bench/smoke.mp4` + `FRAME_OK`。

> ⚠️ `/health` 在**权重加载完成之前**就返回 200,所以切换后的第一单必然是 warmup,**不计时**。
> 本机实测冷/热差距:`turbo_fp8_tp2u2_r50_nfe6` warmup 23.1s → 稳态中位 **18.5s**。
>
> ⚠️ 没打印 `FRAME_OK` 但 `wall=` 正常 → **先查 `which ffmpeg`**(步骤 0),
> `probe.sh:13` 用 `&&` 串联,缺 ffmpeg 时静默不打印,不是引擎的问题。

一次完整基准(warmup 1 + timed 3 取中位,固定 seed):

```bash
bash $B/scripts/bench_matrix.sh turbo fp8 tp2u2 50 6
```

> ⚠️ **首次启动某个组合可能直接报 `SERVER_DIED`,重跑那一格即可。**
> `outputs/bench_results.tsv` **实际是 10 行:7 行 OK + 3 行首跑 SERVER_DIED**
> (`turbo_bf16_tp4_r40` / `turbo_fp8_tp2u2_r50` / `turbo_bf16_tp2u2_r40`,
> 见 `logs/matrix.log`;重跑成功的记录在 `logs/matrix2.log` / `bench_b0.log` / `bench_tp2u2.log`)。
> **上面推荐给你跑的 `turbo fp8 tp2u2 50 6` 正是死过一次的那一格。**
> 死因未定位:`logs/vllm_turbo-fp8-tp2u2-r50.log` 尾部是干净的
> `Shutdown signal received / Application shutdown complete`,
> `grep -iE 'OutOfMemory|CUDA error|Traceback'` **0 命中**。见第 8 节已知坑。
>
> ⚠️ SERVER_DIED 那行给的 `see …` 路径**不可信**:`bench_matrix.sh:11` 读的是
> `$B/logs/vllm_<which>-<prec>-<topo>-r<res>.log`,而 `h3_switch_runpods.sh:99` 只写
> `$B/logs/vllm_server.log`(且 `:100` 的 `rotate_log` 会把上一份改名成
> `vllm_server.<时间戳>.log`)。**要看死因去翻 `logs/vllm_server.log` 和它的时间戳轮转副本。**

下表**只列 7 行 OK**(`outputs/bench_results.tsv`,全部 NFE6 除注明外,864×480):

| 组合 | 中位 wall | 三次 | 峰值显存/卡 |
|---|---|---|---|
| `turbo_fp8_tp2u2_r50` | **18.5s** | 19.0/18.4/18.5 | 19212 MiB |
| `turbo_fp8_tp4_r50` | 20.4s | 20.2/20.4/20.5 | 11712–12232 MiB |
| `turbo_bf16_tp2u2_r40` | 20.6s | 20.6/20.6/24.0 | 28430 MiB |
| `turbo_bf16_tp4_r40` | 20.9s | 21.1/20.9/20.8 | 15548 MiB |
| `turbo_bf16_tp4_r50` | 21.1s | 21.1/21.2/20.9 | 18388 MiB |
| `turbo_fp8_tp2u2_r50` (NFE4) | 16.3s | 16.2/16.4/16.3 | 19192 MiB |
| `turbo_fp8_tp4_r50` (NFE4) | 17.4s | 18.2/17.3/17.4 | 13132–13172 MiB |

### 生产配置(最优)

```yaml
engine:    vLLM-Omni 0.26.0(vllm 0.26.0+cu129)+ PR#5910 @1a9b9c2c + stride patch
model:     merged/MiniMax-H3-Turbo-v4s600ema/FL2VA(BF16 merge,引擎在线 FP8)
topology:  TP2 × Ulysses2(--tensor-parallel-size 2 --usp 2),TP 对落 (0,1)(2,3) 同 NUMA
te:        --text-encoder-tp-size 4   # tp2u2 下必须 = 世界大小,传 2 会崩
dlo:       --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 50
attention: CUDNN_ATTN, --enforce-eager
kernels:   VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel   # sm_120
实测:      NFE6 18.5s,峰值 19212 MiB/卡(32607 MiB 卡上余量充裕)
```

---

## 8. 已知坑与注意事项

按"会不会静默毁掉结果"排序。

1. **stride 补丁丢了不会报错,只会出噪声。** 本机补丁是未提交的工作区修改,任何
   `git checkout/reset/stash/switch` 都会抹掉它。
   > ⚠️ **别看日志抬头的 `stride_patch=` 字段 —— 它恒为 `APPLIED`,毫无鉴别力**(见下一条)。
   > 每次动过 `src-vllm-omni` 之后**手工**跑类内判定
   > (`DLO=…/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py`):
   > `sed -n '/class PinnedResidentLayerGroup/,/def offload/p' "$DLO" | grep -c as_strided`
   > —— 返回 **0 就是补丁没了**,正常应 ≥1(本机当前 2)。
2. **两个脚本的补丁探测都是坏的(裸 grep,恒真)。**
   - `setup_runpods.sh:43` —— `||` 分支永不执行,**连 WARNING 都不会打印,从不打补丁**。
   - `h3_switch_runpods.sh:106` —— 同一个裸 grep,**日志抬头的 `stride_patch=` 永远写
     `APPLIED_as_strided`**;补丁被抹掉后它照样写 APPLIED,引擎照样静默产纯噪声。
     **这是本文档最危险的一条:它曾被当作"补丁是否还在"的自证、排障第一现场和停机门禁,全部失效。**

   根因:未打补丁的原始文件本来就在 299(`physical_view`)/ 412(`prefetch_layer`)行各有一处
   `as_strided`(实测 `git show HEAD:… | grep -n as_strided` → 299、412;类内 `grep -c` → 0)。
   对的写法在 `post_setup.sh:19-20`(裸 grep **与** 类内 grep 的与)与
   `post_setup2.sh:13-14`(纯类内判定)。
   **待修**:把 `h3_switch_runpods.sh:106` 与 `setup_runpods.sh:43` 都换成 `post_setup2.sh:13-14`
   那段类内判定。**在修好之前,`stride_patch=` 字段一律当不存在,按第 1 条手工核。**
   **另外:pod 重建时不要只跑 `setup_runpods.sh` 就以为完事,必须接着跑 `post_setup2.sh`
   或手工执行步骤 4。**
3. **`h3_generate.py` 的 runpods 别名 `turbo-lora → vllm-bf16-original-tp4` 是错的**,
   会把你送到基座 BF16 + NFE11。用全名。
4. **引擎内部 30 秒硬超时(未修)。** `diffusion_engine.py:58` 的
   `_ASYNC_OUTPUT_TIMEOUT = 30.0` 卡的是**两次 yield 之间的间隔**,不是整次请求。
   实测本机 BF16 基座 NFE11(~30s 出头)必挂,报 HTTP 500 + 空 message。
   `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800` 管不到它。
   > **"500 + 空 message" 一律去读服务端 traceback,不要从客户端错误反推;
   > 也别把引擎的等待上限当成硬件/模型的能力上限。**
5. **`setup_runpods.sh` 用的 `HF_HUB_ENABLE_HF_TRANSFER` 已弃用**,速度差约 6 倍。
   改 `HF_XET_HIGH_PERFORMANCE=1`。
6. **`post_setup.sh` 卡在 "waiting for pip install" 再也没往下走。**

   **事实(实测)**:`logs/post_setup.log` 三行到此为止 —— `[00:01:30] waiting for base download...`
   / `[01:01:11] base download done: 168G, transformer shards=13` / `[01:01:11] waiting for pip install...`。
   采集时 `ps -eo pid,ppid,lstart,etime,cmd` 全表里**已经没有任何 `post_setup.sh` 进程**;
   唯一还在空转的是一个 2026-08-09 00:01:46 启动的**外部 watcher**
   (`bash -c until grep -qE "POST_SETUP_COMPLETE|FATAL" …/post_setup.log; do sleep 30; done; tail -25 …`),
   已空转 12 小时。(`post_setup2.sh` 绕开了这个等待,merge 是它完成的。)

   **推测(依据:`post_setup.sh:14` 的 `while pgrep -f "pip install"` 没用方括号)**:
   它可能匹配到某个 cmdline 里含 `pip install` 的包装进程而永不退出。
   **本次未取得直接证据** —— 那个空转 watcher 不是 `post_setup.sh`、也不调用 `pgrep`
   (它 `sleep 30`,而 `post_setup.sh:14` 的循环体是 `sleep 20`);而且 `post_setup.sh`
   自身 cmdline 是 `bash …/post_setup.sh`,不含 `pip install`,pgrep 按设计也不匹配自身进程。
   **也不排除脚本是被外层 ssh 会话结束时带走的。** 对照:`post_setup.sh:8` 的
   `pgrep -f "h[f] download MiniMaxAI"` 用了方括号且确实正常通过了。
   > **方括号铁律作为防御性建议保留**:不只适用于 `kill`,也适用于 `wait`/`pgrep`,
   > 写 `'p[i]p install'`。但"自匹配导致卡死"目前是**待验证的假说,不是已证根因**。
7. **tp2u2 的 `--text-encoder-tp-size` 必须 = 4。** 传 2 会在
   `_build_text_encoder_group` 的 `assert cpu_group is not None` 处全崩。
   `h3_switch_runpods.sh` 固定成 `${H3_TE_TP:-4}` 就是为了这个。
8. **`bf16 + tp2u2 + r40` 显存余量只有约 4G**(峰值 28430 MiB / 32607 MiB)。
   往上调 resident(如 r50)有 OOM 风险。不显式给第 4 个参数就用验证过的默认档
   (bf16=40 / fp8=50)。
   > 纸面估算会误判:按 "BF16 TP2 每卡 33G" 算会判"装不下",实测开了 DLO 只有 28.4G。
9. **`:8091` 被 8 种变体复用。** 没有 `run/vllm.variant` 就无法自证服的是哪个 checkpoint,
   而延迟差异小到无法反推。任何绕过 switcher 的启动方式都必须自己写 `.variant`。
10. **`/` 是临时盘,pod 重建即失。** 包括 `/root/hf-cache/token`、`/root/.cache`、
    `/root/.triton`。只有 `/workspace` 活得下来。
11. **56G 可回收空间**:`base/MiniMax-H3/.cache`(34G)+ `loras/…/.cache`(22G),
    全是下载残留。
12. **本机没有 ComfyUI / SGLang / flash-attn / sageattention / xformers。**
    `scripts/` 里那些 `launch_comfy*.sh`、`install_*backends*.sh`、`test_sglang_*.py`
    是因为"项目根 = 同一个 git 工作区"才存在的,在本机跑不了。
    `test_h3_schedule.py` 会自动降级为只测 vLLM。
13. **benchmark 数字必须声明冷热态。** 冷启动约 **175–180s** 到 `/health` 200
    (实测 `launch 09:35:25` → `Starting vLLM API server` 09:38:18 = 173s);
    同一组合 warmup 23.1s vs 稳态 18.5s,差 25%。引擎侧计时协议 =
    **固定 seed + warmup 1 + timed 3 取中位**(引擎无图缓存,
    不需要 ComfyUI 那套换 seed 破缓存的做法)。
14. **步数语义**:引擎 `num_inference_steps = NFE + 1`(N 个 sigma 点 → N−1 次 forward)。
    换算常数集中在 `scripts/run_fl2va_vllm.py` 的 `NFE_STEP_OFFSET = 1`。
    升级 vllm-omni 后必须重跑 `scripts/test_h3_schedule.py` 确认语义没变。
15. **ssh 相关**:在 Mac 上写命令时双引号里的 `~` 会被本地 shell 展开成 `/Users/ning`,
    远端路径一律用单引号或 `\$HOME`。远端长任务用 `setsid nohup … < /dev/null &` 分离。
16. **HF token 在 `/root/hf-cache/token` 与 `/root/.cache/huggingface/token`(各 37 字节)。**
    值不入库,来源见 `credentials/HF.md`。两个 `.git/config` 的 remote 都是匿名公开 URL,
    没有嵌 token,也没配 credential helper —— 这个状态要保持。
    > 顺带:两个 HF repo 实测都是**公开、非门控**,token 只影响限速/配额,不是硬前提。
17. **`bench_matrix.sh` 首次启动某组合可能报 `SERVER_DIED`,日志里没有 OOM / CUDA error,
    直接重跑那一格即可 —— 原因未定位。** `outputs/bench_results.tsv` 10 行里有 3 行是这种
    (`turbo_bf16_tp4_r40` / `turbo_fp8_tp2u2_r50` / `turbo_bf16_tp2u2_r40`,见 `logs/matrix.log`);
    三格重跑后全部通过(`logs/matrix2.log` / `bench_b0.log` / `bench_tp2u2.log`),
    第 7 节表里的 7 行 OK 就是重跑后的结果。
    实测 `logs/vllm_turbo-fp8-tp2u2-r50.log` 尾部是干净的
    `Shutdown signal received / Application shutdown complete`,
    `grep -iE 'OutOfMemory|CUDA error|Traceback'` **0 命中**。
    > **待修**:`bench_matrix.sh:11` 读的 `$LOG` = `logs/vllm_<which>-<prec>-<topo>-r<res>.log`,
    > 而 `h3_switch_runpods.sh:99` 只写 `logs/vllm_server.log`(`:100` 的 `rotate_log` 把上一份
    > 改名成 `vllm_server.<时间戳>.log`)。**两者不是同一个文件**,所以 SERVER_DIED 行 `:19` 的
    > `grep` 死因必然落空,给出的 `see …` 路径也指向一个 switcher 从不写的文件。
    > 排障请直接看 `logs/vllm_server.log` 及其轮转副本。
18. **`merge_script_sha256` 这一项对不上是预期的,别当成合并出错。** 两机 manifest 记的
    `b034c41b…` 对应的脚本版本**已不在任何机器上、也从未进过 git 历史**;仓库现版本
    (runpods / 6000a / Mac 三处一致)是 `70cdfe9d…`。详见第 5 节与第 7 节步骤 7 的门禁表。
    > **待修(二选一)**:找回 `b034c41b…` 那份脚本并提交、在文档里钉死它的 commit;
    > 或接受现状,永久把该字段排除在门禁之外(本文档现在采用后者)。
19. **"三机权重位级等价"目前只是高置信推测,缺一个真 pin。** `base_transformer_index_sha256`
    只能证明张量名/分片/总字节一致 —— 该文件里**没有任何分片摘要**(实测只有
    `metadata.total_size` + 535 条 `weight_map`)。
    > **待补**:采集并跨机比对 13 个 `transformer/model-000*-of-00013.safetensors` 的 sha256
    > (base 与 merged 各一份),写进 `merge_manifest` 与第 5 节表格。
    > 这是 62G 全盘读,挑两台机器都空闲的时候做。

---

## 9. 本文档的采集方式

- **采集日期**:2026-08-09(pod 时区 UTC,采集窗口 11:31–11:40 UTC)。
- **采集方式**:从 Mac 用 `ssh 5090-Runpods` 分 6 批执行只读脚本
  (`ssh 5090-Runpods 'cat > /tmp/collectN.sh' < 本地脚本 && ssh 5090-Runpods 'bash /tmp/collectN.sh' > 本地输出`),
  覆盖:硬件/系统 → Python 环境与包 → 源码/补丁/权重 → 磁盘明细/模型元数据 → 装机日志与安全核对 →
  下载日志。另跑了一条到 `popos-6000a` 的只读命令做 merge 溯源交叉验证。
  **全程未修改 pod 上任何文件、未装卸包、未停起服务**(采集期间有一个我们自己的
  `turbo-fp8-tp4-r50` 服务在跑,原样保留)。
- **重新采集**:把本文档每张表脚注里的命令重跑一遍即可;最省事的一组是

  ```bash
  ssh 5090-Runpods 'nvidia-smi; nvidia-smi topo -m; nvidia-smi topo -p2p r; lscpu; free -g; df -h /workspace; cat /etc/os-release; uname -a'
  ssh 5090-Runpods '/workspace/h3/env/bin/python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.device_count())"'
  ssh 5090-Runpods 'cd /workspace/h3/src-vllm-omni && git rev-parse HEAD && git status --short && sed -n "/class PinnedResidentLayerGroup/,/def offload/p" vllm_omni/diffusion/offloader/distributed_layerwise_backend.py | grep -c as_strided'
  ssh 5090-Runpods 'cat /workspace/h3/merged/MiniMax-H3-Turbo-v4s600ema/merge_manifest.json'
  ssh 5090-Runpods '/workspace/h3/env/bin/pip freeze --all' > doc/machines/locks/runpods-5090x4.env.pip-freeze.txt   # 记得补回抬头注释
  ```

- **配套清单**:`doc/machines/locks/runpods-5090x4.env.pip-freeze.txt`(唯一环境,无 conda 故无 yml)。
- **本次相对历史档案新增/修正的事实**(供更新总档案时取用):
  ① HF snapshot sha 本机为 `bfc8ed03…` 而非 `b3c7290e…`,但 transformer index sha256 三机相同 →
  snapshot sha 不能当权重 pin;② `env_cu130_unusable` 已删;③ base 135G(FL2VA 本体)vs 168G(含缓存);
  ④ 22G LoRA 误下载残留仍在盘上,另有 34G base 缓存残留;⑤ stride 补丁在本机是**未提交的工作区修改**
  (与 5090 的分支固化不同);⑥ 补上 Turbo LoRA 完整 sha256 与 LoRA revision 全长值;
  ⑦ 补上本机 CPU 型号/NUMA 划分/网卡(档案未记);
  ⑧ **`setup_runpods.sh:43` 与 `h3_switch_runpods.sh:106` 的补丁探测都是恒真的裸 grep** ——
  已取得实证(原始文件 299/412 行本来就有 `as_strided`,类内 0 处);
  因此**日志抬头的 `stride_patch=` 字段不可信**,这一条推翻了旧档案把它当自证的用法。
  ⑨ `post_setup.sh` 卡死一事,**只有"日志停在 waiting for pip install + 进程已不在"是事实,
  "pgrep 自匹配"仍是未证实的推测**(旧稿把它写成了实证,已更正)。
  ⑩ `merge_script_sha256` 记的脚本版本已不存在、不可复现,已从验收门禁里移除。
  ⑪ `base_transformer_index_sha256` 不含分片摘要,**不能证明位级等价**;真 pin(分片 sha256)待补。
  ⑫ 上游 `MiniMaxAI/MiniMax-H3` 在采集当天 10:05 又前进到 `6818f6c3…`(一天内动了两次),
  两个 HF repo 均为公开非门控。
