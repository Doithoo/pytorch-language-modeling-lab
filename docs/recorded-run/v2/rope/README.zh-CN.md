# RoPE Positional Encoding 对照

这是与 learned positional embedding 相同数据、seed、模型规模、optimizer 和训练预算的 RoPE 结构对照。两者都使用 PyTorch SDPA attention。

- Kaggle: <https://www.kaggle.com/code/yashowhoo/pytorch-language-modeling-lab-gpu-rope>
- Kernel version: 1
- positional encoding: `rotary`
- attention implementation: `sdpa`
- Source snapshot: `cb95eb650b3ce22b3cd1e81e274e48f4c66a2125c5b709d2544927814eafce72`，141,791 bytes
- Manifest identity: `3059ee14b15d07394999ce01c0fec50398207999c13efbeea92dba8da2ca12b0`

| 指标 | Learned + SDPA | RoPE + SDPA |
| --- | ---: | ---: |
| 参数量 | 10,763,136 | 10,664,832 |
| 最佳 step | 3,250 | 2,500 |
| Valid perplexity | 3.793969 | **3.548246** |
| Test perplexity | 3.818914 | **3.570483** |
| Test top-1 | 0.590230 | **0.606417** |
| Test top-5 | 0.871391 | **0.877328** |
| Test bits/character | 1.946640 | **1.848921** |
| Token/s | 58,921.9 | 50,160.6 |
| Peak CUDA memory | 1.66 GiB | 1.80 GiB |
| Training seconds | 666.0 | 782.4 |

本次单 seed 结果显示 RoPE 在 Tiny Shakespeare 上获得明显质量提升，但吞吐略低、显存略高，且单次运行不足以形成统计显著性结论。该结果应作为新的实验基线，后续需要多 seed 或更大语料确认泛化趋势。

- [`config.yaml`](config.yaml)
- [`environment.json`](environment.json)
- [`metrics.csv`](metrics.csv)
- [`evaluation/summary.json`](evaluation/summary.json)
- [`provenance.json`](provenance.json)
- [`training-curves.png`](training-curves.png)

checkpoint 二进制不提交 Git，请从上述 kernel 输出下载并按 provenance hash 校验。
