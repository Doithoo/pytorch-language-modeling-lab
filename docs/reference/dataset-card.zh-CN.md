# 数据集说明：Tiny Shakespeare

参考实验使用 `karpathy/char-rnn` 仓库分发的 Tiny Shakespeare：

- URL：`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- SHA-256：`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
- 下载脚本：`scripts/download_data.py`
- 用途：用于模型训练和对照的小型 next-token 语料，不是通用语言模型基准

下载器会在准备前校验 hash。仓库不提交原始语料和 checkpoint；再分发前应自行确认上游来源和许可证边界。

语料规模小、仅英文、文学体裁明显。结果不能外推到现代网页文本、多语言、指令跟随、事实性、安全性或生产性能。
