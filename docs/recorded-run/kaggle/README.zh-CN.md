# Kaggle runner 目录

此目录可以直接交给 `kaggle kernels push`：

- `kernel-metadata.json`：kernel id、GPU、Internet 和入口配置。
- `run_kaggle.py`：由 `scripts/kaggle/build_snapshot.py` 生成的自包含 runner。

runner 内嵌 `src/`、`configs/kaggle_char_small.yaml`、`pyproject.toml` 和 README 的确定性 ZIP 与 SHA-256。它会在 `/kaggle/working/project` 解包源码，安装项目，执行环境 preflight，下载并校验 Tiny Shakespeare，然后依次运行：

```text
prepare-data -> inspect-data -> train -> evaluate -> generate -> summarize
```

修改源码或 Kaggle 配置后必须重新生成 runner：

```bash
uv run python scripts/kaggle/build_snapshot.py
```

不要手工编辑生成文件中的 base64 payload。完整提交、日志、续训和下载步骤见 [Kaggle 指南](../../guides/kaggle.zh-CN.md)。
