"""Download the pinned Tiny Shakespeare corpus used by the Kaggle reference run."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def download(url: str, destination: Path, expected_sha256: str, *, overwrite: bool = False) -> Path:
    if destination.exists() and not overwrite:
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual == expected_sha256:
            return destination
        raise FileExistsError(f"existing file has unexpected SHA-256: {destination}")
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"download SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha256", default=DEFAULT_SHA256)
    parser.add_argument("--output", type=Path, default=Path("data/raw/tiny-shakespeare.txt"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = download(args.url, args.output, args.sha256, overwrite=args.overwrite)
    print(f"downloaded {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
