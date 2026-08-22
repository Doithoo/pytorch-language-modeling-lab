# 数据目录

- `raw/`：原始语料，仅保留小型 fixture 或下载说明。
- `interim/`：规范化文本、重复检测结果和 tokenizer 训练输入。
- `processed/`：token ids、窗口或 shard。
- `manifests/`：数据身份和统计信息。

大文件和可再生产物不提交 Git；来源、许可证、版本和 SHA-256 写入 manifest 或对应文档。
