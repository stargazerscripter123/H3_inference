# 主题: 仓库化与跨机代码同步

范围: 把散在 Mac + 三台 GPU 机的四份 `scripts/` 收敛成一个 git 仓库
(`https://github.com/stargazerscripter123/H3_inference.git`,**公开**),
并让每台机器的项目根**就是**仓库工作区,`git pull` 即更新。

权威结论: FINAL.md。改 `.gitignore` / 分发流程 / 清洗门禁后同步更新。

**本仓库公开** —— 任何新增内容入库前必须过清洗门禁(密钥正则 + 红线素材检查),
门禁脚本与基线见 FINAL.md。`credentials/` 与 `data/` 永久排除。
