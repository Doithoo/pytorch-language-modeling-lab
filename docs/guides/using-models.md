# 使用已发布模型

The recorded runs publish checkpoint hashes and download instructions without committing large binaries. Download a checkpoint from Kaggle, verify its SHA-256, then use the embedded tokenizer for generation.

## Download

```bash
mkdir -p kaggle-output
kaggle kernels output yashowhoo/pytorch-language-modeling-lab-gpu \
  --file-pattern 'artifacts/kaggle-char-small/checkpoints/best.pt' \
  -p kaggle-output
```

Verify the readable checkpoint against `docs/recorded-run/v2/provenance.json`:

```bash
shasum -a 256 kaggle-output/artifacts/kaggle-char-small/checkpoints/best.pt
```

For the SDPA checkpoint, use:

```bash
kaggle kernels output yashowhoo/pytorch-language-modeling-lab-gpu-sdpa \
  --file-pattern 'artifacts/kaggle-char-small-sdpa/checkpoints/best.pt' \
  -p kaggle-output-sdpa
shasum -a 256 kaggle-output-sdpa/artifacts/kaggle-char-small-sdpa/checkpoints/best.pt
```

Expected hashes are listed in [`docs/recorded-run/v2/provenance.json`](../recorded-run/v2/provenance.json) and [`docs/recorded-run/v2/sdpa/provenance.json`](../recorded-run/v2/sdpa/provenance.json).

## Generate

Generation only needs the checkpoint because char/BPE tokenizer payloads are embedded in checkpoint schema v2:

```bash
uv sync --locked --extra dev
uv run lm generate \
  --checkpoint kaggle-output/artifacts/kaggle-char-small/checkpoints/best.pt \
  --prompt "The " \
  --max-new-tokens 200
```

Use `--device cpu` on machines without CUDA. Use `--no-cache` to compare with the full-forward path; fixed-seed cache/no-cache parity is covered by the test suite.

## Evaluate

Perplexity evaluation also needs the matching prepared manifest and token files. Downloading a checkpoint alone is sufficient for generation, but not for reproducing test loss. Prepare the same Tiny Shakespeare manifest with the same config and verify that its manifest identity matches the checkpoint's recorded identity before running:

```bash
uv run lm evaluate \
  --checkpoint kaggle-output/artifacts/kaggle-char-small/checkpoints/best.pt \
  --manifest data/manifests/tiny_shakespeare/manifest.json \
  --split test
```

Do not compare token-level perplexity across different tokenizers without also considering bits-per-character or bits-per-byte.
