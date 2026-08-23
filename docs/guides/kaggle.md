# Kaggle GPU 训练

Kaggle GPU 是正式 small 模型训练和 published run 的主要环境；本地 CPU 用于测试、数据检查和 dry run。

提交前执行：

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
make test
uv run mkdocs build --strict
uv run lm train --config configs/char_tiny.yaml --dry-run --device cpu
uv run python scripts/kaggle/build_snapshot.py
```

修改 `docs/recorded-run/kaggle/kernel-metadata.json` 中的 Kaggle 用户名后提交：

```bash
uv tool install kaggle
kaggle auth login
kaggle kernels push -p docs/recorded-run/kaggle
```

runner 会记录源码快照、数据 hash、环境、训练指标、checkpoint 和生成结果。中断时下载 `last.pt`，下一次任务必须显式提供它；不要依赖上一次 `/kaggle/working`。
