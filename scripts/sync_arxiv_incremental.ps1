# 手动执行 arXiv 增量更新。
# 该脚本与前端“更新数据”按钮使用同一套后端逻辑：
# 读取上次成功更新时间，抓取该时间到当前时间之间 8 个目标分区的新提交论文，
# 写入数据库后自动为新增论文生成 embedding 并追加到本地向量库。

Set-Location (Split-Path $PSScriptRoot -Parent)
conda run -n paper-embed-rec python -m backend.app.cli sync-arxiv-incremental
