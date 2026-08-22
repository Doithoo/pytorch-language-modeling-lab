# 语言模型流程

自回归语言模型学习条件概率：

```text
P(x1, x2, ..., xn) = Π P(xt | x1, ..., x(t-1))
```

项目把每个 token 窗口变成：

```text
input_ids = [BOS, x1, x2, ..., x(T-1)]
targets   = [x1,  x2, x3, ..., xT]
```

模型对每个位置输出整个词表的 logits `[B,T,V]`。交叉熵只统计非 padding target；评估把每个 batch 的 loss 按有效 token 数加权，避免短 batch 与长 batch获得相同权重。

数据流程严格按以下顺序：

```text
source validation -> NFKC normalization -> document split
-> fit tokenizer on train -> encode each document -> window dataset
```

先切分再训练 tokenizer，防止 valid/test 的字符或子词统计进入训练阶段。窗口不跨文档边界，短文档用 pad 补齐，loss 忽略 pad。

训练使用 train 更新参数，valid 选择 `best.pt`，test 只在训练结束后评估固定模型。生成样例是定性证据，不能替代 token loss 和 perplexity。
