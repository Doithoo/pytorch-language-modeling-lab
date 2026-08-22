# 指标

## Token cross-entropy

模型对每个非 padding target 计算交叉熵。评估器累计 `batch_loss * valid_tokens` 再除以总有效 token，因此不同 padding 比例的 batch 不会被同等加权。

## Perplexity

```text
perplexity = exp(mean token cross-entropy)
```

越低表示模型给真实下一个 token 分配的平均概率越高。它不衡量事实性、安全性或长文本一致性。不同 tokenizer 改变 token 单位，因此字符级和 BPE perplexity 不应脱离上下文直接排名。

## 资源指标

`metrics.csv` 记录 optimizer step、train/valid loss、valid perplexity、学习率、累计 token、耗时、token/s 和 CUDA 峰值显存。最终 `run.json`/`kaggle-run-summary.json` 记录参数量、最佳 loss、test 指标和 checkpoint hash。

train loss 是自上次验证以来训练 token 的加权平均；valid 用于选模；test 只在训练结束并固定 best checkpoint 后计算。
