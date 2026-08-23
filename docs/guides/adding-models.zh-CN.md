# 添加模型

新模型必须保持统一契约：输入 `[B,T]`，输出 `[B,T,V]` logits；传入 target 时返回 loss 和有效 token 数；支持 cache 时保持 cache/no-cache 语义一致。

实现放在 `src/language_model/models/`，在 registry 中显式注册，从 `ModelConfig` 重建，不能把 Python 模型对象写入 checkpoint。

需要增加 shape、loss、causal、非法配置、保存/加载和 cache parity 测试，并同步更新模型清单、配置参考、教程和中英文文档。
