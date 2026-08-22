# Scripts

Core behavior lives in `src/language_model/`; scripts only orchestrate external workflows.

- `download_data.py`: download pinned Tiny Shakespeare and verify SHA-256.
- `plot_metrics.py`: plot train/validation loss from `metrics.csv`.
- `kaggle/build_snapshot.py`: create the self-contained runner referenced by Kaggle metadata.

```bash
uv run python scripts/download_data.py
uv run python scripts/kaggle/build_snapshot.py
uv run python scripts/plot_metrics.py artifacts/<run>/metrics.csv
```

All download sources and hashes must be explicit. Generated runners embed a deterministic source archive hash.
