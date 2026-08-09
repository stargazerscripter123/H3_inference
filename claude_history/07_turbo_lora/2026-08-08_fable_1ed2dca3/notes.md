# 2026-08-08 fable 1ed2dca3 — Turbo LoRA 融入 vLLM/SGLang（首个实施 session）

## 目标

按批准的计划（`~/.claude/plans/okay-project-…misty-jellyfish.md`，含 research
feedback 修正）在 6000a 落地 Turbo LoRA：merge 主线 + SGLang dynamic 辅线，
NFE 语义、三层 merge 校验、Stage A 质量快筛。

## 做了什么（时间序）

1. **侦察修正 research note**（doc/turbo_lora.md）：
   - LoRA 键名已是引擎原生 fused 命名（`blocks.N.attn.qkv_proj.lora_A.weight`），
     无需任何转换；alpha 无 metadata → 约定 alpha=rank（scale 1.0，README 明示）。
   - v4_step600_ema 的 canonical 配方 = video shift 12 / audio shift 3（引擎默认）；
     "audio shift 6" 只属旧 ckpt500 实验。
   - 步数语义：引擎 `steps=N` → N−1 forward；`steps=NFE+1` 时 sigma 网格与作者
     generate.py 逐点相同（test_h3_schedule.py 三方验证 ≤8e-8）。
   - vLLM TP4 runtime LoRA 对 fused 层（qkv/fc1）warning 后跳过 → vLLM 只走 merged。
   - SGLang 2D fused lora_B 切片 bug 上游已修（914644e81c9b, PR #33875），
     cherry-pick 为本地分支 `turbo-lora-backport`（commit 4c28e24）。
2. **Gate G0**：下载 v4_step600_ema + 4step_ema_ckpt850（revision afc0346，
   sha256 5f3a626c… / 5a6eeba1…）到 `/home/isaac/Data/h3_weights/loras/`；
   check_turbo_lora.py 259 对/全 BF16/shape 全过；客户端 run_fl2va_{vllm,sglang}.py
   加 `--nfe/--flow-shift/--audio-flow-shift/--dry-run` + manifest 输出，默认
   payload 与原版逐字节一致（本地 diff 验证）；原版备份 `*.orig_20260808`。
3. **Gate G1（merge）**：merge_turbo_lora.py（fp32 累加、qkv ΔW 逆 reorder、
   `.building`+fsync+`.complete` 原子输出、manifest、‖ΔW‖ 报告）。三层校验全过：
   - L1 位级重算 535/535（259 改 + 276 未改）
   - L2 独立 oracle：Comfy 单文件 BF16 == reorder(HF 磁盘) **逐位相等**（6 qkv + 5 平层）
   - L3 单层前向 oracle worst cos 0.999999 / relL2 9.7e-4
   - ΔW 最大在末段 blocks 43-49（rel ≤0.0036）
   - merged 目录 `/home/isaac/Data/h3_weights/MiniMax-H3-Turbo-v4s600ema/`（+62G，盘余 284G）
4. **vLLM merged 烟测**（h3_switch.sh 新 case `vllm-turbo [bf16|fp8]`）：
   ToS 标准对 864×480 seed0，forwards≡NFE（server log 数）：
   - BF16: NFE4=25.1s* / NFE6=26.2s / NFE8=33.5s（3.30 s/it，~46G/卡）
   - FP8: NFE4=26.5s* / **NFE6=23.1s** / NFE8=28.9s（2.78 s/it，~38G/卡）
   - *为切服务后首单含 compile；对照 base FP8 12步=37.4s（1.62×）
   - 帧目检（NFE4/6/8 × BF16/FP8）：无拖影/崩坏/量化伪影，NFE6/8 比 12 步 base 略锐
5. **SGLang merged 烟测**（case `sglang-turbo`）：两个坑及修复：
   - 根目录需完整 HF 布局 → 原根所有顶层项 symlink 进 merged 根
   - registry 按路径 basename 短名匹配（须为 `minimax-h3`）→ 别名目录
     `turbo_v4s600ema_alias/MiniMax-H3 -> merged root`
   - 结果 1376×768 seed0：NFE4=59.6s / NFE6=72.1s / NFE8=93.1s
     （9.9 s/it，decode 6.5s；对照 base 12步=126.2s → NFE6 1.75×）
6. **SGLang dynamic 烟测**（case `sglang-lora`，backport 分支）：
   - TP 单测过（test_sglang_lora_slice.py：2D fused 切片 + 前向数学 + 3D 回归，TP1/2/4）
   - LoRA 命中 **259/266 层**（7 层未覆盖 = LoRA 未打的 patch/time/out 投影，精确）
   - STANDARD 格式零转换；unmerged strength 1.0；`/v1/list_loras` 正常
   - NFE6=75.2s（10.74 s/it，dynamic 开销 ~8%）；**峰值 VRAM 47-48G 贴顶**
     → dynamic 定位=研究/A-B 工具，非生产
7. **Stage A**（进行中）：8 case ToS 素材（sniper/bridge/scope/shout/holo/sky/
   fight/robot，镜头切分检测选窗，864×480 cover-crop，`inputs/stage_a_final/`）+
   7 个新 prompt（`workflows/stage_a/`）；ckpt850 vs v4s600 A/B（sglang dynamic
   热切换，fight/scope/bridge × NFE4/6）后台运行中。
8. Mac `scripts/h3_generate.py`：新增 profile `turbo-lora`（vllm-turbo fp8, NFE6
   默认）与 `sglang-turbo`（NFE6），`--nfe` 透传；旧 profile 语义不变（py_compile 过）。

## 踩坑（新增台账项）

- SGLang 本地目录加载 H3：根 model_index.json 的 `_class_name` 是
  MiniMaxH3**Modular**Pipeline（registry 无此类 → 跌 diffusers 回退崩溃）；
  native 识别靠 `KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS` 按 basename 短名
  == `minimax-h3` 匹配 → 本地目录必须叫 `MiniMax-H3`（别名 symlink 目录解决）。
- sglang 服务器 `/health` 早于权重加载可用（vLLM 同理），首单必为 warmup。
- ssh 单引号内嵌 heredoc 的 while-read 会被截字符 → 远端循环用显式列表/上传脚本。
- 上游 cookbook（PR #33875 文档部分）写 "set num_inference_steps: 4 or 8"，与
  代码实测语义（N 点→N−1 forward）不符，且偏好 ckpt500；以作者 README（v4s600ema）
  与我们的 schedule 单测为准。

## 后半程补记（2026-08-09 凌晨，同 session）

9. **ckpt850 A/B**（sglang dynamic 热切，12 单）：fight（极端运动）NFE4 下
   ck850 保留动态姿态/运动模糊明显好于 v4s600（v4 把动作静态化或融毁）。
10. **turbofp8 主链 36 单零失败**：Stage A 8case×NFE4/6/8 + audio ablation +
    a2a 1344×768 + 正式计时。**bench: NFE6=23.5s(23.5/23.5/23.6)、
    NFE4=17.9s(×3)**；a2a 768P NFE6: vLLM 69.7-70.6s ≈ SGLang 72.1s。
11. **base 链 40 单零失败**：base FP8 NFE4/6/8/11（NFE11=37.7s 复现旧档 ✓）+
    BF16 NFE20 oracle（~76s）。归因：base 少步单帧被端点约束托住，但 YDIF
    运动能量前载后僵（bridge 2.51→0.99），turbo 全程持续且高 17-25%。
12. **audio shift ablation**：shift6 无增益（RMS 持平/-1.6dB，无异常）→ 弃。
13. **fight@NFE4(v4, 864×480) 确认失败**（融毁），NFE6 边界；其余 7 case
    NFE6 帧级全优。
14. **e2e**：Mac CLI `--profile turbo-lora` 全链路过（NFE6，回传成功）。
    另发现横幅仍打 "steps 12"（仅显示瑕疵，未修——该文件与 5090 session 并行
    编辑中，避免冲突）。
15. 归档：doc/turbo_lora_results.md 定稿、FINAL.md 建档、gallery 增 Turbo LoRA
    区块与大运动对比、06_infra 台账 9-14 条、auto-memory 更新。

## 未竟 / 交接

- **Stage B 盲评需用户**：finalist × 12-20 case × seed 0/1/2；通过后考虑把
  `turbo-lora` 转正为 6000a 默认 turbo。
- 5090 侧换 merged 目录（05 主题 DLO 产线可直接受益）未做。
- audio shift 弃用前留 1 条人工抽听确认（outputs/ablation_audioshift6/）。
