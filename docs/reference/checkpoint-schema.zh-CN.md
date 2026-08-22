# Checkpoint 结构

schema version 1 使用 `torch.save` 写入纯数据和 tensor，并始终通过 `torch.load(..., weights_only=True)` 加载。

| 字段 | 内容 |
| --- | --- |
| `config` | 完整解析配置与实际词表大小 |
| `manifest_identity` | 训练数据身份 |
| `tokenizer_fingerprint` | tokenizer 身份 |
| `tokenizer_payload` | 独立生成所需 char/BPE 内容 |
| `model_state` | 模型 state dict |
| `optimizer_state` | AdamW 状态 |
| `scheduler_state` | warmup/cosine 状态 |
| `scaler_state` | AMP GradScaler 状态 |
| `trainer_state` | global step、最佳 loss、token 数、耗时 |
| `rng_state` | Python、NumPy、PyTorch 和 CUDA RNG |

`best.pt` 是验证 loss 最低的模型，供最终评估和生成；`last.pt` 是最近训练状态，供中断续训。续训同时校验 manifest、tokenizer 和模型配置。

checkpoint 不保存 Python 模型对象或任意可执行工厂。生成可以只使用 checkpoint 内嵌 tokenizer；数据评估仍需要对应 manifest 和 `.bin` 文件。
