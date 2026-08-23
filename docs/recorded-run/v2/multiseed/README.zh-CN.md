# RoPE 多随机种子复现实验

为确认 seed 42 的 RoPE 优势不是单次随机波动，项目在相同 5,000-step 预算下补充 seed 7 和 seed 123。每个 seed 内，learned + SDPA 与 RoPE + SDPA 使用相同数据 split、tokenizer、训练顺序 seed、模型规模和 optimizer；唯一结构变量是 positional encoding。

## 配对结果

| Seed | Learned test PPL | RoPE test PPL | 相对变化 | Learned top-1 | RoPE top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 3.818914 | **3.570483** | -6.51% | 0.590230 | **0.606417** |
| 7 | 3.943149 | **3.680229** | -6.67% | 0.580554 | **0.596624** |
| 123 | 3.815546 | **3.581002** | -6.15% | 0.589127 | **0.608673** |

汇总：

| 指标 | Learned + SDPA | RoPE + SDPA |
| --- | ---: | ---: |
| Test perplexity 均值 | 3.859203 | **3.610572** |
| Test perplexity 样本标准差 | 0.072719 | 0.060554 |
| 配对差值均值（RoPE - learned） |  | **-0.248631** |
| 配对差值样本标准差 |  | 0.014189 |

三个 seed 的 test perplexity 和 top-1 都同方向改善，RoPE 相对 perplexity 降幅稳定在 6.15%-6.67%。这比单 seed 结果更有说服力，但仍只代表 Tiny Shakespeare、小模型和当前训练预算，不应外推为所有语料与模型规模的普遍结论。

## 性能取舍

RoPE + SDPA 的训练吞吐约 47.9k-50.3k token/s，learned + SDPA 约 58.9k-60.0k token/s。RoPE 在当前实现中获得更好质量，但每 step 计算更慢；后续优化方向是预计算 cos/sin cache，而不是改变实验结论。

## 证据

- [`summary.json`](summary.json)：三 seed 配对结果、均值、标准差、kernel 和 checkpoint hash。
- [`seed7/sdpa/`](seed7/sdpa/) 与 [`seed7/rope/`](seed7/rope/)：seed 7 配置、manifest、指标、评估和环境。
- [`seed123/sdpa/`](seed123/sdpa/) 与 [`seed123/rope/`](seed123/rope/)：seed 123 的对应证据。
- Seed 42 证据位于 [`../sdpa/`](../sdpa/) 与 [`../rope/`](../rope/)。

四个新增 kernel 的 version 1 因 runner manifest 路径硬编码而在训练前失败；修复后 version 2 完成。失败版本不进入统计结果。
