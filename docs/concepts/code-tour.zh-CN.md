# 代码导览

推荐阅读顺序：

1. `tokenization/char.py`：最短的 text/ids 双向映射。
2. `data/dataset.py`：窗口如何右移成输入和 targets。
3. `models/bigram.py`：最小 `[B,T,V]` 模型契约。
4. `models/attention.py`：QKV、causal mask 和 KV cache。
5. `models/transformer.py`：embedding、decoder block、loss。
6. `evaluation/metrics.py`：token-weighted loss 与 perplexity。
7. `training/trainer.py`：梯度累积、AMP、验证和保存。
8. `training/checkpoint.py`：可恢复状态与 `weights_only` 加载。
9. `generation/`：采样过滤和增量生成。
10. `training/train.py` 与 `cli.py`：模块编排和用户命令。

边界原则：CLI 不解析语料、不构造 attention、不直接操作 checkpoint 内部字段；Kaggle runner 不包含模型逻辑；checkpoint 不序列化 Python 模型对象；评估不隐式重新训练 tokenizer。
