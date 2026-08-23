# 实验比较

公平实验应固定来源、split、manifest、tokenizer、模型规模、优化器、训练预算和 prompt，只改变一个研究变量。

```bash
uv run lm compare-runs --run artifacts/run-a --run artifacts/run-b
```

当前 CLI 会记录 manifest 和 tokenizer identity；不同 tokenizer 时不再生成 test perplexity 排名，而使用 BPC/BPB 作为跨 tokenizer 参考。多 seed 实验应报告均值和样本标准差，不应把一次 Tiny Shakespeare 运行描述为普遍架构结论。
