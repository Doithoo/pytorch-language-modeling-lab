# 数据与 Manifest

manifest schema version 1 记录：

- 数据集名称、源文件相对路径/字节数/SHA-256。
- NFKC 规范化版本、去空/去重统计。
- 文档级 split ratios、seed 和每个 split 的 document ids。
- context length、stride、window mode 和 `uint32` dtype。
- tokenizer 名称、词表大小、fingerprint 和特殊 token ids。
- 每个 `.bin` 的 SHA-256、token/window 数和 document offsets。
- 去除创建时间与绝对路径后计算的稳定 manifest identity。

`.bin` 是连续 `uint32` token 数组。offsets 保证窗口停留在单个文档内；dataset 通过 memmap 只读取当前窗口。短窗口补 pad，target loss 忽略 pad。

`inspect-data` 验证 manifest identity、tokenizer fingerprint、词表大小和 token 文件 SHA-256。训练和评估默认再次验证，避免在错误数据上声明指标。

更换规范化、split、tokenizer、上下文、stride 或源字节都会改变身份。`created_at`、机器绝对路径和产物位置不会改变数据身份。
