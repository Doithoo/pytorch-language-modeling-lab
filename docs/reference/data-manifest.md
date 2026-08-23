# Data Manifest

Manifest schema version 1 records source file hashes, normalization, split policy, tokenizer identity, context and stride, token files, document offsets, and a stable identity.

The identity excludes creation time, absolute source root, and output location. It changes when source bytes, normalization, split seed or ratios, tokenizer, context length, stride, or window policy changes.

Prepared files are contiguous `uint32` token arrays. Offsets keep windows within one document. Short windows are padded, and padding targets are ignored by loss.

`inspect-data` validates manifest structure, identity, tokenizer fingerprint, vocabulary size, and token-file SHA-256. Training and evaluation verify the data again before reporting metrics.
