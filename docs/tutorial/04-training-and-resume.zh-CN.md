# 训练与恢复

先在 CPU 执行一个 batch 的 dry run：

```bash
uv run lm train --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json \
  --dry-run --device cpu
```

完整运行会写入配置、manifest、环境、`run.json`、`metrics.csv`、tokenizer、checkpoint 和 evaluation 目录。

`best.pt` 由验证 loss 最小的 checkpoint 选出，用于最终评估；`last.pt` 保存 optimizer、scheduler、scaler、sampler 和 RNG 状态，用于中断恢复：

```bash
uv run lm train --config configs/char_tiny.yaml \
  --manifest data/manifests/tiny_corpus/manifest.json \
  --resume artifacts/char-tiny-transformer/checkpoints/last.pt
```

恢复前会校验模型、训练、manifest、tokenizer 和 data-order contract。
