# 评估与生成

使用匹配的 manifest 评估 checkpoint：

```bash
uv run lm evaluate \
  --checkpoint artifacts/<run>/checkpoints/best.pt \
  --manifest data/manifests/<dataset>/manifest.json \
  --split test --device cpu
```

checkpoint schema v2 内嵌 tokenizer，因此生成不需要当前目录的词表：

```bash
uv run lm generate \
  --checkpoint artifacts/<run>/checkpoints/best.pt \
  --prompt "ROMEO:" --max-new-tokens 200 --device cpu
```

支持 greedy、temperature、top-k、top-p、EOS、seed 和 KV cache。不同 tokenizer 的 token-level perplexity 不应直接排名，应同时查看 BPC/BPB。
