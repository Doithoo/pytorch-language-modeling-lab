# 使用已发布模型

项目不提交大体积 checkpoint，而是在 recorded run 中保存下载地址和 SHA-256。下载后先校验 hash，再使用 checkpoint 内嵌的 tokenizer 生成文本。

## 下载 readable checkpoint

```bash
mkdir -p kaggle-output
kaggle kernels output yashowhoo/pytorch-language-modeling-lab-gpu \
  --file-pattern 'artifacts/kaggle-char-small/checkpoints/best.pt' \
  -p kaggle-output
shasum -a 256 kaggle-output/artifacts/kaggle-char-small/checkpoints/best.pt
```

SDPA checkpoint：

```bash
kaggle kernels output yashowhoo/pytorch-language-modeling-lab-gpu-sdpa \
  --file-pattern 'artifacts/kaggle-char-small-sdpa/checkpoints/best.pt' \
  -p kaggle-output-sdpa
shasum -a 256 kaggle-output-sdpa/artifacts/kaggle-char-small-sdpa/checkpoints/best.pt
```

当前三 seed 研究中表现最稳定的是 RoPE checkpoint：

```bash
kaggle kernels output yashowhoo/pytorch-language-modeling-lab-gpu-rope \
  --file-pattern 'artifacts/kaggle-char-small-rope/checkpoints/best.pt' \
  -p kaggle-output-rope
shasum -a 256 kaggle-output-rope/artifacts/kaggle-char-small-rope/checkpoints/best.pt
```

期望 hash 见 [`docs/recorded-run/v2/`](../recorded-run/v2/README.zh-CN.md) 下对应实验的 provenance 文件。

## 生成

生成只需要 checkpoint。char/BPE tokenizer payload 已嵌入 checkpoint schema v2：

```bash
uv sync --locked --extra dev
uv run lm generate \
  --checkpoint kaggle-output/artifacts/kaggle-char-small/checkpoints/best.pt \
  --prompt "The " \
  --max-new-tokens 200
```

没有 CUDA 时增加 `--device cpu`。使用 `--no-cache` 可以对比完整 forward 路径；固定 seed 下两条路径的 parity 已由测试覆盖。

## 评估

perplexity 评估还需要匹配的 manifest 和 token 文件。仅下载 checkpoint 足以生成文本，但不能复现 test loss。准备 Tiny Shakespeare 时使用相同配置，并在评估前确认 manifest identity 与 recorded run 一致：

```bash
uv run lm evaluate \
  --checkpoint kaggle-output/artifacts/kaggle-char-small/checkpoints/best.pt \
  --manifest data/manifests/tiny_shakespeare/manifest.json \
  --split test
```

不同 tokenizer 的 token-level perplexity 不能直接比较，应同时观察 bits-per-character 或 bits-per-byte。
