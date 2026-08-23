# Scripts

Core behavior lives in `src/language_model/`; scripts orchestrate external workflows.

- `download_data.py`: download the pinned Tiny Shakespeare source and verify SHA-256.
- `plot_metrics.py`: plot training and validation loss from `metrics.csv`.
- `kaggle/build_snapshot.py`: build the self-contained runner referenced by Kaggle metadata.

```bash
uv run python scripts/download_data.py
uv run python scripts/kaggle/build_snapshot.py
uv run python scripts/plot_metrics.py artifacts/<run>/metrics.csv
```

All download sources and hashes must be explicit. Generated runners embed a deterministic source archive hash.
