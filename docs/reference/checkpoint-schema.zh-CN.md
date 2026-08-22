# Checkpoint 结构

schema version 2 使用 `torch.save` 写入纯数据和 tensor，并始终通过 `torch.load(..., weights_only=True)` 加载。schema 1 checkpoint 仍可用于评估和生成，但不能宣称支持严格可复现续训。

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
| `trainer_state` | global step、micro step、epoch/batch offset、最佳 loss、token 数、耗时 |
| `data_order` | dataset size、batch size、seed、shuffle 和可恢复 sampler 算法版本 |
| `rng_state` | Python、NumPy、PyTorch 和 CUDA RNG |

`best.pt` 是验证 loss 最低的模型，供最终评估和生成；`last.pt` 是最近训练状态，供中断续训。续训同时校验 manifest、tokenizer、模型、完整 optimizer/scheduler 配置和 data-order contract。训练器按 `micro_step` 重建每个 epoch 的 permutation，因此同一配置下中断恢复与连续训练的模型参数一致。

checkpoint 不保存 Python 模型对象或任意可执行工厂。生成可以只使用 checkpoint 内嵌 tokenizer；数据评估仍需要对应 manifest 和 `.bin` 文件。
