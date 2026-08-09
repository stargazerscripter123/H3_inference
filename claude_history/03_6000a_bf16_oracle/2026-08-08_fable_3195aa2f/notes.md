# 03 · 6000a 4卡 BF16 Oracle(2026-08-05)

## 目标
零量化 BF16 全精度参照(doc 的量化诊断 Gate),4×48GB 全驻留。

## 做了什么
- BF16 权重(Comfy 单文件版 DiT 66.3G + TE 51.5G)下到**第二块 NVMe**
  `/home/isaac/Data/h3_weights/`(根盘 92% 满),symlink 回 ComfyUI models
- 停用 isaac 的 kreaid ComfyUI(GPU0/1, :8188/:8189,**用户批准**;
  原命令在 `/home/isaac/workdir/kreaid/`,恢复时照抄)
- 我们的 ComfyUI 重启为 4 卡可见(:8288,`launch_comfy_6000a_4gpu.sh`)
- ComfyUI-MultiGPU **DisTorch2** 节点:DiT compute cuda:2 + vvram 35G donor cuda:3;
  TE compute cuda:0 + vvram 30G donor cuda:1;VAE cuda:1
  (6000 Ada 专业卡 P2P 全互通,5090 上崩的路径这里不触发)
- runner 加 `--dit-compute/--dit-vvram/--dit-donor/--te-compute/...` 参数

## 关键数据
- wall 400s(冷含 118G 加载),denoise 10.49s/it ×30(≈ INT8 的 3.2×)
- 实际 VRAM:gpu0 47.6 / gpu1 7.4 / gpu2 47.3 / gpu3 0.7G
  (donor 分配未按预期落 gpu3,DiT 溢出走了内存流式 —— 不影响正确性,慢一点)
- **同 seed 下 pruned INT8 与 BF16 输出高度一致,832×480 无系统性质量差
  → pruned INT8 可作该分辨率生产档位**(doc Gate 3 结论)

## 产出
- CLI `--profile bf16`(仅 6000a);gallery 三方对比(输入/INT8/BF16)

## 未竟
- donor 精调(expert_mode_allocations)可再压延迟 —— 低优先级,oracle 用途够了
