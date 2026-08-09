# RunPods 4×5090 TP4 部署与提速结果(2026-08-09)

对 `doc/TP4_5090.md` research note 的执行与实测修正。机器:RunPods 云 pod,
4×RTX5090 32.6G、1007G RAM(note 说 574G,实际更多)、`/workspace` 503G 持久卷
+ `/` 1TB 临时盘、Ubuntu 24.04 / Python 3.12,无 conda、无 ComfyUI。

## 一、机器实测底数(先于任何 benchmark)

| 项目 | 实测 | 影响 |
|---|---|---|
| GPU 拓扑 | GPU0/1→NUMA0,GPU2/3→NUMA1,跨对 `SYS` | TP2 分组必须 (0,1)(2,3) |
| **GPU P2P** | `nvidia-smi topo -p2p r/w` **全对 `CNS`(不支持)** | **所有 NCCL 集合走 host 中转,通信是本机主要成本** |
| `/workspace` 写入 | 664 MB/s | 不是下载瓶颈 |
| `/`(overlay,本地 NVMe) | 2.3 GB/s | 仅放可再生缓存 |
| CPU | 128 线程 | 充裕 |

### 通信量推演(决定实验矩阵优先级)
864×480×124 帧的 DiT 激活约 5.0万 token × 7168 dim × 2B ≈ **720MB**;每个 block
两次 row-parallel all-reduce × 50 blocks = 每次 forward ~100 次 720MB 集合通信。
ring all-reduce 每卡流量 ∝ 2(N−1)/N ⇒ **TP4 通信量是 TP2 的 1.5×**,而每卡算力只减半。
在无 P2P 的机器上,这意味着 TP4 不必然最快 —— 故本轮把 **FP8 + TP2×U2**(通信最省)
列为与 BF16 TP4 同等重要的候选,而不是 note 里的"先 TP4 再考虑 TP2×U2"。

### 显存可行性(DiT 权重,不含激活/TE/VAE)
| 精度 | TP4/卡 | TP2/卡 |
|---|---|---|
| BF16(66G) | 16.5G ✓ | 33G ✗ **超 32G,BF16 TP2×U2 不可全常驻** |
| FP8(33G) | 8.3G ✓ | 16.5G ✓ |

## 二、部署踩坑(pod 重建必读)

1. **HF 下载速度**:`HF_HUB_ENABLE_HF_TRANSFER=1` 已被 huggingface_hub 弃用,
   走退化路径只有 ~28MB/s;换 **`HF_XET_HIGH_PERFORMANCE=1`** 后峰值 247MB/s
   (广域网仍有大幅波动,145G 实际约 40 分钟)。
2. LoRA 仓不要整仓拉(22 文件含多个 checkpoint,白下 22G):
   `--include "minimax_h3_turbo_v4_step600_ema.safetensors"`。
3. **vLLM-Omni PR#5910 的新 head `1a9b9c2c` 仍带 resident-repoint 丢 stride 的 bug**
   (2026-08-09 核实,`PinnedResidentLayerGroup.load()` 仍是 `.view(shape)`)。
   必须打 `scripts/pr5910_resident_stride_fix.patch`,否则 FP8 路径输出纯噪声
   —— 根因与证据链见 `claude_history/08_5090_tp2/FINAL.md`。
4. 一键恢复:`scripts/setup_runpods.sh`(Mac 项目内),把 `h3_switch_runpods.sh`
   `bench_matrix.sh` `probe.sh` `run_fl2va_vllm.py` `merge_turbo_lora.py` 推到
   `/workspace/h3/scripts/` 即可。

3. **驱动决定 CUDA 大版本,必须选对 vLLM wheel**(本轮最大的坑):
   pod 驱动 570.195.03 → 只支持 **CUDA 12.8**。PyPI 的 `vllm==0.26.0` 拉的是
   torch 2.11.0+**cu130**,启动即 `RuntimeError: The NVIDIA driver on your system
   is too old (found version 12080)`。
   - 正确做法:装 vLLM 官方发布的 **`+cu129`** wheel。CUDA 12.9 与 12.8 同属大版本 12,
     minor-version 兼容(12.x 家族要求 driver ≥525 <580);CUDA 13.0 要求 ≥580.65。
   - **不存在 cu128 wheel**(vLLM 从未发过 `+cu128` tag,官方文档那句"also provide
     CUDA 12.8 binaries"是过期文本 —— 别去找)。
   ```bash
   pip install https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl \
     --extra-index-url https://download.pytorch.org/whl/cu129
   ```
   - 若 pod 驱动 ≥580(如本地 popos-5090 的 580.173.02),普通 `vllm==0.26.0` 即可。
4. venv 不可重定位:如果先建 `env129` 再 `mv` 成 `env`,`env/bin/*` 的 shebang 仍指旧路径,
   报 "No such file or directory"(指的是解释器不是脚本)。要么原地建,要么 sed 修 shebang
   + `pyvenv.cfg`。
5. 一键恢复:`scripts/setup_runpods.sh`(Mac 项目内,已含 cu129 逻辑),把
   `h3_switch_runpods.sh` `bench_matrix.sh` `run_matrix*.sh` `run_fl2va_vllm.py`
   `merge_turbo_lora.py` `pr5910_resident_stride_fix.patch` 推到 `/workspace/h3/scripts/`。

## 三、Benchmark 结果

固定:864×480、124 帧、Turbo v4 step600 EMA merged、CUDNN_ATTN、eager、
引擎侧固定 seed 0、warmup 1 + timed 3 取中位。全部配置均先抽帧过质量关再计时。

| 精度 | 拓扑 | resident | NFE | **中位** | 三次 | 峰值 VRAM/卡 |
|---|---|---:|---:|---:|---|---:|
| BF16 | TP4×U1 | 40 | 6 | 20.9s | 21.1/20.9/20.8 | 15.5G |
| BF16 | TP4×U1 | 50 | 6 | 21.1s | 21.1/21.2/20.9 | 18.4G |
| BF16 | TP2×U2 | 40 | 6 | 20.6s | 20.6/20.6/24.0 | 28.4G |
| FP8 | TP4×U1 | 50 | 6 | 20.4s | 20.2/20.4/20.5 | 12.2G |
| **FP8** | **TP2×U2** | 50 | 6 | **18.5s** | 19.0/18.4/18.5 | 19.2G |
| FP8 | TP4×U1 | 50 | 4 | 17.4s | 18.2/17.3/17.4 | 13.1G |
| **FP8** | **TP2×U2** | 50 | 4 | **16.3s** | 16.2/16.4/16.3 | 19.2G |

### 三条与 note 预判不同的结论

1. **TP2×Ulysses2 快于 TP4(18.5 vs 20.4s,−9%)**,与第一节的通信量推演一致:
   Ulysses 只在 qkv 处做 all-to-all,避开了 TP4 每层两次跨 4 卡 all-reduce。
   note 把 TP4 列为"第一条必须跑的",在有 P2P 的机器上成立,**在这台无 P2P 的机器上不成立**。
2. **residency 冲到 50 没有收益**(TP4 BF16:r40 20.9s vs r50 21.1s,差异在噪声内)。
   1007G RAM 下剩余 10 个块的 PCIe 流式传输被计算完全掩盖 —— note 里"最终目标
   dlo_resident_layers=50"不必追求,r40 反而省 3G/卡。
3. **BF16 TP2×U2 装得下**(28.4G/卡 @ r40):权重按 33G/卡 估算会得出"放不下"的结论,
   但 DLO 只常驻 40/50 个块,实测可行(仅比 TP4 快 0.3s,不值得为它牺牲 9G/卡余量)。

### 与其它机器对照(同素材同画布)

| 机器 | 产线 | NFE | warm |
|---|---|---:|---:|
| **RunPods 4×5090** | vLLM FP8 TP2×U2 Turbo | 6 | **18.5s** |
| 同上 | 同上 | 4 | **16.3s** |
| 6000a 4×RTX6000Ada | vLLM FP8 TP4 Turbo | 6 | 23.5s |
| popos 2×5090 | vLLM FP8 TP2 Turbo | 6 | 29.0s |
| popos 2×5090 | ComfyUI TeaCache(有损) | — | 35s |

4 张 5090 比 4 张 RTX6000 Ada 快 **1.27×**(23.5→18.5s),尽管后者有完整 P2P ——
sm_120 的算力优势盖过了无 P2P 的通信劣势。

## 四、FP4 的结论

**不做,无工程路径**:vllm-omni 没有 H3 的 FP4 权重路径(`--quantization` 支持
fp8/int8/mxfp4/mxfp4_dualscale/bitsandbytes/inc,其中 mxfp4 系列是 NPU-only);
comfy-kitchen 的 NVFP4 是 ComfyUI 专用格式,vLLM/SGLang 都不能加载(用户此前已明确弃用)。
FP8 已把 DiT 压到 8.3G/卡,显存不再是这台机器的约束(峰值仅用掉 19G/32G),
再降精度收益是算力而非容量,而 sm_120 的 FP4 GEMM 在 vLLM 侧尚无 H3 接线。

## 五、生产配置

```yaml
host: runpods (4x RTX 5090, driver 570 / CUDA 12.8)
engine: vLLM-Omni 0.26.0+cu129 + PR#5910 @1a9b9c2c + pr5910_resident_stride_fix.patch
model: MiniMax-H3 Turbo v4s600ema merged (BF16 merge -> 引擎在线 FP8)
precision: global online FP8
topology: TP2 x Ulysses2   # TP pair (0,1)(2,3) 同 NUMA;text-encoder-tp-size=4
dlo: no_allgather, resident_layers 50
attention: CUDNN_ATTN, enforce-eager
nfe: 6 (balanced) / 4 (fast)
resolution: 864x480, 124 frames
```

入口:`h3_switch_runpods.sh turbo fp8 tp2u2 50`;
Mac CLI `--host runpods --profile vllm-fp8-turbo-tp2u2 [--nfe 4]`。
E2E 实测:启动(切换+就绪)196s + 推理 22.7s;音频 aac 立体声正常。
