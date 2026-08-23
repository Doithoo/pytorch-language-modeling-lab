# Dataset Card: Tiny Shakespeare

The reference experiment uses the Tiny Shakespeare text distributed by the `karpathy/char-rnn` repository:

- URL: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
- Retrieval: `scripts/download_data.py`
- Role: a small next-token corpus for model development and comparison, not a general language benchmark

The downloader verifies the hash before preparation. The repository does not commit the raw corpus or model checkpoints. Users are responsible for checking the upstream source and applicable public-domain or repository licensing terms before redistribution.

The corpus is small, literary, and English-only. Results should not be generalized to modern web text, multilingual modeling, instruction following, factuality, safety, or production performance.
