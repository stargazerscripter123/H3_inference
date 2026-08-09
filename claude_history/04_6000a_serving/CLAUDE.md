# 主题: 6000a vLLM-Omni / SGLang 4卡 serving

范围: 官方权重(HF 格式)/ 两栈源码安装 / h3_switch.sh 互斥切换 / benchmark V1/V2/A3 / CLI profile turbo|vllm|sglang(6000a 语义)。
权威结论: FINAL.md,数据档 doc/speedup_6000a_results.md。
关键: 三条 serving 产线互斥占 4 卡;SGLang API 锁 768 短边;所有安装坑的修法在 FINAL.md,重装照抄。
