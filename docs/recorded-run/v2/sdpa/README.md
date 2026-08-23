# SDPA Comparison

This run compares PyTorch scaled dot-product attention with the readable reference implementation under the same seed-42 configuration and data identity.

| Implementation | Test perplexity | Throughput | Peak memory |
| --- | ---: | ---: | ---: |
| Readable | 3.827989 | 41,735.7 token/s | about 2.64 GiB |
| SDPA | **3.818914** | **58,921.9 token/s** | **about 1.66 GiB** |

SDPA is materially faster and uses less memory in this environment. The small quality difference should not be interpreted as a model-quality claim because the implementation comparison is primarily about attention execution.

Evidence: [`provenance.json`](provenance.json), [`config.yaml`](config.yaml), [`environment.json`](environment.json), [`metrics.csv`](metrics.csv), and [`evaluation/summary.json`](evaluation/summary.json).
