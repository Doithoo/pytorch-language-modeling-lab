# Kaggle v3 Recorded Run

This is the completed `0.2.2` GPU validation run. It submitted the updated runner as Kaggle kernel version 4 on a Tesla T4.

- Kaggle: <https://www.kaggle.com/code/yashowhoo/pytorch-language-modeling-lab-gpu>
- Kernel version: 4
- Release: `0.2.2`
- Source snapshot: `c9b552ba7752eb271670576da7b94160a563fa58594f7a7ba84bcd36e70c607c` (154,393 bytes)
- Manifest identity: `22ab79d0c225b6b824e1f21a9a230036d0e1c880efeebd1a12975b71528553b3`
- Tokenizer fingerprint: `2f21f34c3e7146a043097c9d5c3e97a16de8d280f7e3a97fee031300ae5baf3e`

## Result

| Metric | Value |
| --- | ---: |
| Updates | 5,000 |
| Best validation step | 2,750 |
| Best validation perplexity | **3.793904** |
| Test perplexity | **3.827989** |
| Test top-1 accuracy | 0.586632 |
| Test top-5 accuracy | 0.870946 |
| Effective training target tokens | 39,229,485 |
| Throughput | 42,466.1 token/s |
| Training elapsed time | 924.0 seconds |
| Peak CUDA memory | 2,829,815,296 bytes, about 2.64 GiB |

The metrics are consistent with the historical readable baseline. This run confirms the `0.2.2` source snapshot, provenance metadata, packaging, and documentation release; it is not a quality improvement claim.

## Environment

Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, Tesla T4, compute capability 7.5. Kaggle exposed two GPUs; the project used `cuda:0` only.

## Files

- [`provenance.json`](provenance.json): source, corpus, checkpoint, environment, and result hashes.
- [`data-summary.json`](data-summary.json): lightweight corpus and split summary.
- [`config.yaml`](config.yaml): resolved training configuration.
- [`metrics.csv`](metrics.csv): all validation records.
- [`evaluation/summary.json`](evaluation/summary.json): final valid/test metrics.
- [`evaluation/prompts.jsonl`](evaluation/prompts.jsonl): fixed prompt generation.
- [`runner-v4.py`](runner-v4.py): exact self-contained runner submitted to Kaggle.

Checkpoint binaries are intentionally excluded from Git. Download them from kernel version 4 and verify the SHA-256 values in `provenance.json`.
