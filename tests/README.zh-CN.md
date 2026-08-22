# 测试策略

测试分为：

- `unit/`：配置、char/BPE、数据窗口、causal 模型、KV cache、采样和 CLI。
- `contract/`：Markdown 链接、包元数据、示例执行、Kaggle metadata 和自包含 runner 构建。
- `integration/`：合成 JSONL 的 `prepare -> dry run -> train -> evaluate -> generate -> resume`。
- `fixtures/`：只存放未来需要的极小静态输入；当前语料由 pytest 临时生成。

```bash
uv run pytest -W error::DeprecationWarning
```

测试不联网、不下载语料、不需要 GPU。随机测试显式设置 seed；cache 与 no-cache 使用数值容差比较。大型 Kaggle 训练不进入 CI，完成证据由 `docs/recorded-run/` 单独记录。
