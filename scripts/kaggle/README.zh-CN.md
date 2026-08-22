# Kaggle 辅助脚本

此目录预留本地侧 Kaggle 工具：

- `build_snapshot.py`：收集允许进入 kernel 的源码并计算 SHA-256。
- `collect_output.py`：校验下载产物并抽取 recorded-run 轻量证据。
- `check_metadata.py`：在提交前检查 metadata、runner、配置和数据源。

Kaggle 内部入口放在 `docs/recorded-run/kaggle/run_kaggle.py`，便于与 metadata 和已记录运行一起审计。核心训练逻辑始终保留在 `src/language_model/`。
