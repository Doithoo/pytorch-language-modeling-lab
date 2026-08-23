# 贡献指南

这是一个可复现的因果语言模型项目。贡献范围包括数据、tokenizer、模型、训练、评估、推理和实验记录。

提交前运行：

```bash
uv sync --locked --extra dev
make check
pre-commit run --all-files
```

行为变化必须补充针对性测试。集成测试保持离线、CPU-only，并使用合成语料。不要提交原始大语料、生成 token 文件、凭证或 checkpoint。新的 recorded run 必须保留准确源码快照、数据身份、环境、配置、指标和 checkpoint hash。

修改公开契约时，同时更新中英文参考文档；若 checkpoint 或 manifest 兼容性变化，必须增加迁移说明。
