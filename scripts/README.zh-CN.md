# 脚本目录

核心逻辑位于 `src/language_model/`，脚本只编排外部流程：

- `download_data.py`：下载固定 Tiny Shakespeare 并校验 SHA-256。
- `plot_metrics.py`：从 `metrics.csv` 绘制 train/valid loss。
- `kaggle/build_snapshot.py`：生成 metadata 引用的自包含 Kaggle runner。

```bash
uv run python scripts/download_data.py
uv run python scripts/kaggle/build_snapshot.py
uv run python scripts/plot_metrics.py artifacts/<run>/metrics.csv
```

下载来源和 hash 必须显式记录；runner 内嵌确定性源码压缩包和摘要。
