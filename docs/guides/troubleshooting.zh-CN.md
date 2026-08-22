# 排障

## 找不到语料

`prepare-data` 只递归读取 `.txt` 和 `.jsonl`。JSONL 每行必须是包含字符串 `text` 的对象；可选 `id` 必须唯一。

## 目标目录已存在

数据准备和新训练默认拒绝覆盖。数据使用 `prepare-data --overwrite`；训练应修改 `runtime.run_name`。只有续训时使用 `train --resume <last.pt>` 进入已有目录。

## manifest 或 tokenizer 不匹配

不要手工编辑 `manifest.json`、tokenizer 文件或 `.bin`。任何改变 token 序列的操作都应重新运行 `prepare-data --overwrite`。checkpoint 会拒绝不匹配的 manifest identity 或 tokenizer fingerprint。

## CUDA 不可用

本地使用 `--device cpu`。Kaggle 页面确认 Accelerator 已启用；runner 的 `preflight` 会在下载和训练前检查 CUDA、GPU 型号、compute capability、显存和磁盘。

## 显存不足

依次降低 `train.batch_size`、`model.context_length`、`model.d_model` 或层数，再增加 `gradient_accumulation_steps` 保持有效 batch。改变上下文长度后必须重新准备数据，因为它属于 manifest 身份。

## loss 是 NaN

先关闭 AMP，降低学习率，确认 batch 中存在非 padding targets，并检查语料不是空文本。训练器默认执行梯度裁剪；若仍失败，使用 tiny 配置和 CPU dry run 定位。

## perplexity 很大

perplexity 是 `exp(token loss)`，未训练模型很大是正常现象。比较实验时必须使用相同 tokenizer 和 split；不同词表下的 perplexity不能直接视为完全等价。

## Kaggle 任务中断

下载 `artifacts/<run>/checkpoints/last.pt`，在下一次任务中作为明确输入提供，再使用 `--resume`。Kaggle 上一次会话的 `/kaggle/working` 不会永久保留。
