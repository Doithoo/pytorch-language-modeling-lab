# 配置参考

配置加载顺序是 YAML，再应用重复的 `--set section.field=value`。未知 section/field 会失败。

## `data`

- `corpus_dir`：`.txt`/`.jsonl` 根目录。
- `processed_dir`、`manifest_dir`：token 文件和身份元数据目录。
- `dataset_name`：隔离不同语料/tokenizer 实验。
- `tokenizer_name`：必须与 `tokenizer.name` 一致。
- `context_length`、`stride`：窗口长度和起点间隔。
- `split_ratios`：三个正数，和必须为 1。
- `num_workers`：DataLoader worker 数。
- `split_strategy`：`document_count` 或 `character_balanced`；后者按规范化字符数近似平衡 split。
- `group_by_source`：按 source 分组切分，适合多书籍、多文件语料；单一 source 不应开启。

## `tokenizer`

`name` 是 `char` 或 `bpe`。BPE 必须指定至少 32 的 `vocab_size`。四个特殊 token 顺序固定为 pad、BOS、EOS、UNK。

## `model`

`name` 是 `bigram` 或 `transformer_decoder`。Transformer 使用 `n_layers`、`n_heads`、`d_model`、`d_ff`、`dropout`；`d_model` 必须整除 `n_heads`。`positional_encoding` 可选 `learned` 或 `rotary`；RoPE 要求每个 attention head 的维度为偶数。`attention_implementation` 可选 `readable` 或 `sdpa`；前者用于教学，后者调用 PyTorch scaled dot-product attention。`vocab_size` 通常为 null，由 manifest 解析。

## `train`

`max_steps` 是 optimizer updates，不是 micro-batches。有效 batch 是 `batch_size * gradient_accumulation_steps`。还包括 AdamW 学习率/权重衰减/betas、warmup、梯度裁剪、AMP、日志/验证/保存间隔与 `eval_batches`。

## `generation`

包含最大新 token 数、temperature、top-k、top-p、KV cache、seed 和 `prompts`。训练完成后会为每个 prompt 写一行 JSONL。

## `runtime`

`device` 是 auto/cpu/cuda/mps；`output_dir/run_name` 构成运行目录。`deterministic=true` 会启用 PyTorch 确定性算法和固定 CUDA 设置。新训练拒绝写入非空目录。
