# RoPE Comparison

This run compares the seed-42 RoPE Transformer with the learned-position SDPA control under the same Tiny Shakespeare manifest, model size, optimizer, and 5,000-update budget.

- Positional encoding: rotary
- Attention: PyTorch SDPA
- Test perplexity: **3.570483**
- Test top-1 accuracy: **0.606417**
- Peak memory: about 1.79 GiB

RoPE improved the measured result in this paired run while reducing throughput relative to learned positions. The multi-seed replication is the stronger evidence and should be read together with this single-run report.

Evidence: [`provenance.json`](provenance.json), [`config.yaml`](config.yaml), [`environment.json`](environment.json), [`metrics.csv`](metrics.csv), and [`evaluation/summary.json`](evaluation/summary.json).
