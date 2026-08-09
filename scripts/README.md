# scripts/ 索引

47 个文件，分四类。**通用**的三机可用；**机器专属**的只在对应机器有意义（保留是因为
它们记录了各机的真实配置与历史事故，合并只会互相污染）。

命名里的机器代号：`5090` = 2×RTX5090 本地机，`6000a` = 4×RTX6000Ada 本地机，
`runpods` = 4×RTX5090 云机。

## 一、编排（在工作机上跑，不在 GPU 机上跑）

| 文件 | 说明 |
|---|---|
| `h3_generate.py` | 主 CLI。切后端 → 等就绪 → 远端生成 → 回传，分别报告启动与推理耗时。`--list` 看 host × profile 能力矩阵。**机器台账目前在这个文件的 `HOSTS` 字典里**（Phase 2 会外置成 machines.yaml） |
| `h3_eval.py` | 把一台/多台机器的所有 profile 用同一素材跑一遍出对比表。协议 = 1 warmup（不计）+ N timed；**逐遍换 seed**，否则测到的是 ComfyUI 的整图缓存 |

## 二、通用工具与测试（三机可用）

| 文件 | 说明 |
|---|---|
| `merge_turbo_lora.py` | 把 Turbo LoRA 合进官方 BF16 分片。fp32 累加、qkv 逆 reorder、原子输出 + `merge_manifest.json` + `.complete` 契约、三层校验（位级/Comfy oracle/单层前向）。权重根用 `--base` 或 `H3_BASE_FL2VA` 给 |
| `check_turbo_lora.py` | LoRA 文件体检：259 对键、全 BF16、shape 逐表核对 |
| `test_h3_schedule.py` | 三方 sigma 网格一致性（作者解析式 vs vLLM vs SGLang）。引擎源码按 `H3_SRC_ROOT` / `H3_ROOT` 定位；没装 SGLang 的机器自动降级只测 vLLM |
| `test_sglang_lora_slice.py` | SGLang `slice_lora_b_weights` 的 TP1/2/4 单测（2D fused lora_B 的 backport 验证） |
| `test_h3_switch_5090.sh` | switch 脚本的决策表单测（`H3_SWITCH_LIB=1` source 取函数并 stub 掉启动）。逻辑通用，常量是 5090 的 |
| `pr5910_resident_stride_fix.patch` | **必打**。vllm-omni PR#5910 的 `PinnedResidentLayerGroup.load()` 丢 stride，不打会让 FP8+DLO 静默产出纯噪声 |
| `extract_frames.sh` | 从源视频抽首尾帧对（间隔 123 帧 @24fps，cover+crop） |

## 三、产线控制（机器专属：三套互不兼容的 CLI 契约，Phase 2 合一）

| 文件 | 机器 | CLI |
|---|---|---|
| `h3_switch_5090.sh` | 5090 | `comfy \| comfy-tlora \| vllm \| vllm-turbo \| sglang` |
| `h3_switch.sh` | 6000a | `baseline \| vllm [bf16\|fp8] \| vllm-turbo […] \| sglang \| sglang-turbo \| sglang-lora` |
| `h3_switch_runpods.sh` | runpods | `<original\|turbo> <bf16\|fp8> <tp4\|tp2u2> [resident]`（三维正交，最干净） |

三份共有的运维铁律（改任何一份都要保持）：变体追踪（`run/vllm.variant` + `.model`，
共用端口时防静默服错 checkpoint）、**同变体且健康则直接复用不重启**、半死进程清理、
端口占用拒启、`flock` 串行化**且守护进程必须 `9>&-`**（否则 fd 继承会把锁一直持有）、
启动后 `kill -0` 确认存活、日志轮转不截断。

复用那条不只是省事：runpods 冷启动 210s，缺了它每次生成都白付这 210s，而且**测到的
推理耗时会虚高 20%**（重启后首推 22.3s vs 真 warm 18.5s）。2026-08-09 补齐时正是
靠这个差值发现的。

ComfyUI worker 启动器：`launch_comfy.sh`（通用，收 `<gpu> [port] [extra]`）、
`launch_comfy_5090.sh` / `_turbo.sh`(:8189 TeaCache) / `_tlora.sh`(:8190 Turbo LoRA)、
`launch_comfy_6000a_4gpu.sh`(:8288)。

## 四、远端客户端（由编排层调用，一般不手动跑）

| 文件 | 后端 |
|---|---|
| `run_fl2va.py` | ComfyUI（API-format 图提交 + 轮询）。含 Turbo LoRA 节点与 TeaCache 接线 |
| `run_fl2va_vllm.py` | vLLM-Omni `/v1/videos/sync`（multipart，必须走 curl） |
| `run_fl2va_sglang.py` | SGLang `/v1/videos` 异步 + 轮询 + `/content` |

三者都支持 `--nfe`（实际 forward 数）并写 `*.manifest.json` 审计（nfe/steps/shifts/
seed/sigma 指纹/峰值显存/`served_model`）。**引擎侧 `steps = NFE+1`，ComfyUI 侧
`steps == NFE`** —— 换算常数集中在各客户端的 `NFE_STEP_OFFSET`。

## 五、装机与权重（机器专属，一次性）

- 本地机：`setup_h3.sh`、`install_backends.sh` / `install_backends2.sh`(6000a)、
  `install_5090_backends.sh` / `fix_5090_installs.sh`(5090)
- 云机：`setup_runpods.sh`、`post_setup.sh` / `post_setup2.sh` / `fix_cu129.sh`
- 权重：`download_h3.sh`（ComfyUI 单文件，带字节校验+续传）、`download_bf16.sh`(6000a)、
  `dl_fl2va_auth.sh` / `dl_fp8_auth.sh`

⚠️ `setup_runpods.sh` 有三个已知缺陷待修（Phase 2）：stride 补丁探测恒真所以从不打补丁、
用了已弃用的 `HF_HUB_ENABLE_HF_TRANSFER`（应为 `HF_XET_HIGH_PERFORMANCE`，速度差 6 倍）、
依赖尚未推送到机器上的 `scripts/*`。

## 六、基准与实验（机器专属）

`bench_turbo.sh`(5090) / `bench_timed.sh`(6000a) / `bench_matrix.sh` + `run_matrix*.sh`(runpods)
是同一件事的三次独立重写；`smoke_vllm_turbo.sh` / `smoke_sglang_turbo.sh` /
`stage_a_run.sh` / `ab_ckpt850_dynamic.sh`(6000a) 是 Turbo LoRA 的扫描与 A/B；
`tp2_r2_*.sh`(5090/6000a) / `probe.sh`(runpods) 是 TP2 噪声定位那一轮的探针。
新的评测优先用 `h3_eval.py`，这些留作历史参照。
