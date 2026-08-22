# 生成与采样

模型最后位置 logits 先除以 temperature，再经过可选过滤：

- `temperature=0`：greedy，始终选最大 logit。
- `temperature<1`：分布更集中。
- `temperature>1`：分布更平坦。
- `top_k=k`：只保留 logit 最大的 k 个 token。
- `top_p=p`：保留累计概率达到 p 的最小候选集合。

过滤后使用 multinomial 采样。相同 checkpoint、prompt、参数、设备和 seed 应得到可复现结果，但跨 PyTorch/硬件不承诺逐位一致。

生成最多产生 `max_new_tokens`，遇到 EOS 提前停止。prompt 自动添加 BOS，但输出解码默认隐藏特殊 token。

KV cache 减少重复计算，不改变采样分布。`--no-cache` 可用于参考对比：

```bash
uv run lm generate --checkpoint best.pt --prompt "ROMEO:" --no-cache
```

不要把流畅文本等同于事实正确、安全或无偏。这个项目训练的是小型续写模型，不是生产对话系统。
