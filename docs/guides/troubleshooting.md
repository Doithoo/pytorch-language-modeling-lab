# Troubleshooting

## Corpus not found

`prepare-data` searches recursively for `.txt` and `.jsonl`. JSONL rows need a string `text` field, and explicit ids must be unique.

## Existing output

Use `--overwrite` only for intentional data rebuilds. Use a new `runtime.run_name` for a new experiment. Use `--resume` only with a compatible `last.pt`.

## Manifest or tokenizer mismatch

Do not edit generated files. Re-run preparation after changing source text, split settings, tokenizer, context length, or stride.

## CUDA and memory

Use `--device cpu` locally. On GPU, lower batch size, context length, model width, or layer count; increase gradient accumulation to retain an effective batch size. A context-length change requires a new manifest.

## NaN or very large perplexity

Disable AMP and lower the learning rate first. Verify that batches contain valid targets. Perplexity is `exp(token loss)` and is not directly comparable across different tokenizers.

## Interrupted Kaggle run

Download `artifacts/<run>/checkpoints/last.pt` and provide it explicitly to a new task. Kaggle working directories are not durable.
