# RoPE Multi-Seed Replication

The paired seed-42, seed-7, and seed-123 runs compare learned positions with RoPE while holding corpus, tokenizer, model size, optimizer, update budget, and data-order policy constant.

| Seed | Learned + SDPA PPL | RoPE + SDPA PPL | Relative change |
| ---: | ---: | ---: | ---: |
| 42 | 3.818914 | **3.570483** | -6.51% |
| 7 | 3.943149 | **3.680229** | -6.67% |
| 123 | 3.815546 | **3.581002** | -6.15% |

Mean test perplexity is 3.859203 for learned positions and **3.610572** for RoPE. The paired difference is consistent across these three seeds, but this remains a small Tiny Shakespeare study rather than a universal architecture claim.

Machine-readable results are in [`summary.json`](summary.json). Seed-specific configurations, metrics, manifests, and environments are under `seed7/` and `seed123/`; seed 42 evidence is in the sibling `sdpa/` and `rope/` directories.
