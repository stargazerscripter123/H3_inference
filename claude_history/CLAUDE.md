# claude_history — AI session 工作档案(约定)

本目录是 minimax-h3 项目所有 AI session 的工作档案。**任何 AI session 开始工作前先读本文件。**

## 结构

```text
claude_history/
├── CLAUDE.md                 ← 本文件(总约定,不要改动实质规则)
├── _session_summaries/       ← 每个 session 一份跨主题总摘要:<日期>_<model>_<sessionID8>.md
├── <NN>_<subject>/           ← 一级 = 主题(如 01_deploy_smoke, 04_6000a_serving)
│   ├── CLAUDE.md             ← 主题说明:范围、现状指针、主题特有规则
│   ├── FINAL.md              ← ★整理合并的最终结论(跨 session 维护的权威文档)
│   └── <日期>_<model>_<sessionID8>/   ← 二级 = session 原始记录(只归该 session 写)
│       └── notes.md          ← 目标/做了什么/关键数据/产出路径/踩坑/未竟
└── ...
```

## 规则(必须遵守)

1. **原始记录不可侵犯**:session 目录(二级)只归创建它的 session 写;
   其它 session 只读,永不修改/删除。session 目录名含全局唯一 session ID,天然不冲突。
2. **FINAL.md 是共享可变的权威文档**:每次有新原始记录入库后,由当前 session
   把新事实**合并**进 FINAL.md(带日期标注)。只增改结论,不删除仍然有效的历史事实;
   推翻旧结论时写明"X 被 Y 取代(日期+原因)"。
3. **新主题**:开新的一级目录(编号顺延),同时创建该主题的 CLAUDE.md 与 FINAL.md。
4. **接手未竟工作**:读该主题 FINAL.md 与最新 session 的 notes.md,在**自己的**
   session 子目录里开 notes.md 续写,文首注明"继承自 <session>"。
5. 每个 session 结束(或断线归档)时:更新涉及主题的 FINAL.md + 写 `_session_summaries/` 总摘要。

## 项目速查(详见各主题 FINAL.md)

- 两台 GPU 机器:`popos-5090`(2×RTX5090 32G)/ `popos-6000a`(4×RTX6000Ada 48G),
  远端工作根:`~/data/dropbox/CV/h3/`
- Mac 入口:`scripts/h3_generate.py --host {5090|6000a|both} --profile {baseline|turbo|vllm|sglang|bf16}`
- benchmark 档案:`doc/speedup_5090_results.md`、`doc/speedup_6000a_results.md`;可视化:`gallery/index.html`
- **内容红线**(对所有 session 有效):不执行涉及**真实可识别人物的无授权私密影像**的
  生成任务,也不协助把针对这类素材的命令调通。`data/` 下存在此类受限素材,该目录整体
  排除出仓库且不得入库。既往 session 已多次拒绝并被用户接受,该立场持续有效。
  基准素材一律用 Tears of Steel(CC-BY)。
