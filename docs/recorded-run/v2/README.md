# Kaggle Recorded Run v2

This is the first complete run produced by the optimized `0.2.0` codebase. It uses Kaggle kernel version 3 on a Tesla T4, with the readable attention implementation, checkpoint schema v2, deterministic batch ordering, token-weighted evaluation, and the new accuracy/BPC/BPB metrics.

- Kaggle: <https://www.kaggle.com/code/yashowhoo/pytorch-language-modeling-lab-gpu>
- Kernel version: 3
- Release: `0.2.0`
- Source snapshot: `a506d0cb60901264f248a1922368c4713eadc8ebbd4139e71a57999b66a76edf` (140,563 bytes)
- Manifest identity: `3059ee14b15d07394999ce01c0fec50398207999c13efbeea92dba8da2ca12b0`
- Tokenizer fingerprint: `2f21f34c3e7146a043097c9d5c3e97a16de8d280f7e3a97fee031300ae5baf3e`

## Results

| Metric | Value |
| --- | ---: |
| Updates | 5,000 |
| Best step | 2,750 |
| Best validation loss | 1.333396 |
| Best validation perplexity | **3.793904** |
| Test loss | 1.342340 |
| Test perplexity | **3.827989** |
| Test top-1 accuracy | 0.586632 |
| Test top-5 accuracy | 0.870946 |
| Test bits per character | 1.950089 |
| Effective training target tokens | 39,229,485 |
| Final throughput | 41,735.7 token/s |
| Training time | 940.2 seconds |
| Peak CUDA memory | 2,829,815,296 bytes, about 2.64 GiB |

The old recorded run reached validation perplexity 3.786090 and test perplexity 3.815852. The small difference is expected because this run uses a new PyTorch/CUDA environment and a new training implementation. It should be treated as a new baseline, not as a claim of regression or improvement from one run alone.

![Training and validation loss](training-curves.png)

## Readable vs SDPA

The same `0.2.0` configuration was run with PyTorch SDPA as a controlled attention implementation comparison. SDPA completed in 666.0 seconds at 58,921.9 token/s and about 1.66 GiB peak CUDA memory; the readable run took 940.2 seconds at 41,735.7 token/s and about 2.64 GiB. Test perplexity was 3.818914 vs 3.827989, and test top-1 accuracy was 0.590230 vs 0.586632. See the [SDPA comparison record](sdpa/README.md).


The same `0.2.0` configuration was also run with RoPE positional encoding. RoPE reached test perplexity 3.570483 and top-1 accuracy 0.606417 in the seed-42 run, compared with 3.818914 and 0.590230 for learned positions with SDPA. Two additional paired seeds confirmed the direction: mean test perplexity was 3.610572 for RoPE vs 3.859203 for learned positions. See the [three-seed replication](multiseed/README.md) and [RoPE comparison record](rope/README.md).


The `0.2.1` shared autocast-aware RoPE table removed repeated trigonometric work while retaining equivalent quality and baseline-level throughput. See the [cache optimization record](rope-cache/README.md).


- [`manifest.json`](manifest.json): prepared data identity and split metadata.
- [`metrics.csv`](metrics.csv): 20 validation records including top-1/top-5 accuracy.
- [`evaluation/summary.json`](evaluation/summary.json): final valid/test metrics from `best.pt`.
- [`evaluation/prompts.jsonl`](evaluation/prompts.jsonl): fixed prompt output and generation parameters.
- [`comparison.json`](comparison.json): machine-readable readable/SDPA/RoPE comparison under the same data identity.

Checkpoint binaries are not committed. Download `artifacts/kaggle-char-small/checkpoints/best.pt` or `last.pt` from kernel version 3 and verify against [`provenance.json`](provenance.json).
