# 使用自己的数据

支持 `data.corpus_dir` 下的 UTF-8 `.txt` 和 `.jsonl`。

JSONL 每行格式：

```json
{"id": "doc-001", "text": "一段文本"}
```

`text` 必须是字符串；显式 `id` 必须唯一。项目会先按文档切分 train/valid/test，再只用 train 训练 tokenizer。多来源语料可使用 `group_by_source: true` 保持 source 隔离。

每种 tokenizer、context length 和 stride 都应使用独立 dataset name。不要手工编辑 `.bin`、tokenizer 或 manifest，修改来源后使用 `prepare-data --overwrite` 重建。
