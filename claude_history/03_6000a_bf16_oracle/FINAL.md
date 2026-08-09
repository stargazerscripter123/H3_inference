# FINAL · 6000a BF16 Oracle(整理版, 2026-08-08 by 3195aa2f)

## 配置(权威)
CLI `--host 6000a --profile bf16`;ComfyUI :8288 4卡 + DisTorch2:
DiT bf16 66.3G(compute cuda:2, vvram 35 donor cuda:3)+ TE bf16 51.5G(cuda:0, vvram 30 donor cuda:1)+ VAE cuda:1。
BF16 权重在 6000a 第二块 NVMe `/home/isaac/Data/h3_weights/`(根盘勿放大文件,92% 满)。

## 结论(量化诊断)
- 400s/条(10.49s/it);同 seed 与 pruned INT8 在 832×480 无系统性差异
  → **pruned INT8 = 该分辨率生产档位**,BF16 仅作诊断参照
- 已知瑕疵: DisTorch donor 未落 gpu3(溢出走内存流式),正确性无损
- kreaid 服务(GPU0/1)经用户批准停用;恢复: /home/isaac/workdir/kreaid/ 下原命令
