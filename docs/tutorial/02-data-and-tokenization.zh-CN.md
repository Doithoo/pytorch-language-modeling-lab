# 数据与 Tokenizer

数据流水线固定为：

```text
校验来源 -> 规范化 -> 文档切分 -> 只用 train 训练 tokenizer
-> 编码 -> 文档内窗口 -> manifest 与 hash
```

`.txt` 会被解析为段落或非空行；`.jsonl` 每行必须有字符串 `text`，也可以有唯一 `id`。

字符 tokenizer 是确定性的；byte-level BPE 使用独立的 `dataset_name`，避免 manifest 和 token 文件混用：

```bash
uv run lm prepare-data --config configs/bpe_tiny.yaml
uv run lm inspect-data --manifest data/manifests/tiny_corpus_bpe/manifest.json
```

源文件、规范化、split、tokenizer、context length 或 stride 变化都会改变 manifest identity。
