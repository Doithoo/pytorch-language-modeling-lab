# RoPE cos/sin Cache 优化记录

`0.2.1` 将 RoPE cos/sin 从每层每次 forward 的动态三角函数计算改为模型级预计算、非持久 buffer。buffer 不进入 state dict，因此 `0.2.0` checkpoint 可直接加载。

## 受控结果

所有运行使用 seed 42、相同 manifest、模型、SDPA、AMP 和 5,000-step 预算。

| 实现 | Test PPL | Token/s | 训练秒数 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: |
| 动态 RoPE 基线 | 3.570483 | 50,160.6 | 782.4 | 1.80 GiB |
| 每层 cache，version 1 | 3.570488 | 48,994.5 | 801.0 | 1.80 GiB |
| 共享 FP32 cache，version 2 | 3.570501 | 45,737.7 | 858.0 | 1.79 GiB |
| 共享 autocast-aware cache，version 3 | 3.570488 | **50,457.7** | **777.7** | 1.80 GiB |

version 1 因每层重复 index/cast 变慢；version 2 在 AMP 下用 FP32 table 与 FP16 Q/K 相乘，触发类型提升，明显变慢。version 3 每个 model forward 只按当前 autocast dtype 切片一次，并由所有 block 复用。

最终吞吐比动态基线高约 0.59%，可能落在单次运行噪声内。保留该实现的主要理由是移除重复三角函数、共享缓存、保持数学结果和 checkpoint 兼容，而不是宣称显著加速。

- [`optimization.json`](optimization.json)：三次迭代的机器可读结果。
- [`provenance.json`](provenance.json)：accepted version 3 的源码和 checkpoint hash。
- [`config.yaml`](config.yaml)、[`metrics.csv`](metrics.csv)、[`evaluation/summary.json`](evaluation/summary.json)。
- Kernel: <https://www.kaggle.com/code/yashowhoo/pytorch-language-modeling-lab-gpu-rope-cached>
