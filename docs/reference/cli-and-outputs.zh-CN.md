# CLI 与输出

使用 `uv run lm --help` 和 `uv run lm <command> --help` 查看最终参数。

| 命令 | 输入 | 主要输出 |
| --- | --- | --- |
| `prepare-data` | config | manifest 和 `.bin` |
| `inspect-data` | manifest | 校验后的 JSON 摘要 |
| `train` | config、可选 manifest | run 目录、metrics、checkpoint |
| `evaluate` | checkpoint、manifest | JSON 指标 |
| `generate` | checkpoint、prompt | JSON 文本、token ids、参数 |
| `compare-runs` | 多个 run 目录 | 安全的比较结果 |
| `list-models` | 无 | 模型清单 |
| `show-config` | config | 解析后的 JSON 配置 |
| `preflight` | 可选 work dir | 环境信息 |

完整运行包含 `config.yaml`、`manifest.json`、`environment.json`、`run.json`、`metrics.csv`、`tokenizer/`、`checkpoints/` 和 `evaluation/`。
