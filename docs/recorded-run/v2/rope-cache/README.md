# Cached RoPE Optimization

The `0.2.1` implementation moves RoPE cosine and sine tables to a model-shared, non-persistent, autocast-aware cache. It preserves checkpoint state-dict compatibility with the dynamic implementation.

| Version | Implementation | Test perplexity | Throughput |
| ---: | --- | ---: | ---: |
| baseline | Dynamic RoPE | 3.570483 | 50,160.6 token/s |
| 1 | Per-layer cached float16 table | 3.570488 | 48,994.5 token/s |
| 2 | Shared float32 table | 3.570501 | 45,737.7 token/s |
| 3 | Shared autocast-aware table | **3.570488** | **50,457.7 token/s** |

The accepted implementation removes repeated trigonometric work without a meaningful quality change. The measured throughput difference is small and may be within run-to-run noise.

See [`optimization.json`](optimization.json) and [`provenance.json`](provenance.json) for the complete evidence.
