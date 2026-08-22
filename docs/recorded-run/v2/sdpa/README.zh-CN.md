# SDPA Attention 对照运行

这是与 readable attention 同数据、同 seed、同模型和同训练预算的 PyTorch SDPA 对照运行。

- Kaggle: <https://www.kaggle.com/code/yashowhoo/pytorch-language-modeling-lab-gpu-sdpa>
- Kernel version: 1
- attention implementation: `sdpa`
- Source snapshot: `0514eeb1cd334abb6b8ca2272cdc804308c9015083057637493da01a7f1ff7b2`，140,529 bytes
- Manifest identity: `3059ee14b15d07394999ce01c0fec50398207999c13efbeea92dba8da2ca12b0`

| 指标 | Readable | SDPA |
| --- | ---: | ---: |
| 最佳 step | 2,750 | 3,250 |
| Valid perplexity | 3.793904 | 3.793969 |
| Test perplexity | 3.827989 | **3.818914** |
| Test top-1 | 0.586632 | **0.590230** |
| Test top-5 | 0.870946 | 0.871391 |
| Test bits/character | 1.950089 | **1.946640** |
| Token/s | 41,735.7 | **58,921.9** |
| Peak CUDA memory | 2.64 GiB | **1.66 GiB** |
| Training seconds | 940.2 | **666.0** |

这不是严格的统计显著性结论，单个 seed 和单次训练不足以证明模型质量提升；它用于展示 fused attention 对吞吐和显存的影响，并验证 readable/SDPA 在数值和训练链路上的兼容性。

- [`config.yaml`](config.yaml)
- [`environment.json`](environment.json)
- [`metrics.csv`](metrics.csv)
- [`evaluation/summary.json`](evaluation/summary.json)
- [`provenance.json`](provenance.json)
- [`training-curves.png`](training-curves.png)

checkpoint 二进制不提交 Git，请从上述 kernel 输出下载并按 provenance hash 校验。
