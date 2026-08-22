"""Build the self-contained Kaggle script used by kernel-metadata.json."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "recorded-run" / "kaggle" / "run_kaggle.py"
INCLUDE_ROOTS = ("src", "configs/kaggle_char_small.yaml", "pyproject.toml", "README.md")

RUNNER_TEMPLATE = '''"""Generated self-contained Kaggle runner. Rebuild with scripts/kaggle/build_snapshot.py."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SNAPSHOT_SHA256 = "__SNAPSHOT_SHA256__"
SNAPSHOT_BASE64 = "__SNAPSHOT_BASE64__"
CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
WORKING = Path("/kaggle/working")
PROJECT = WORKING / "project"


def event(phase: str, status: str, **details: object) -> None:
    print(json.dumps({"phase": phase, "status": status, **details}), flush=True)


def run(*command: str) -> None:
    event("command", "started", command=list(command))
    subprocess.run(command, cwd=PROJECT, check=True)
    event("command", "completed", command=list(command))


def main() -> None:
    archive = base64.b64decode(SNAPSHOT_BASE64)
    actual = hashlib.sha256(archive).hexdigest()
    if actual != SNAPSHOT_SHA256:
        raise RuntimeError(f"source snapshot hash mismatch: {actual}")
    PROJECT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        bundle.extractall(PROJECT)
    event("snapshot", "completed", bytes=len(archive), sha256=actual)
    run(sys.executable, "-m", "pip", "install", "-e", ".")
    run(sys.executable, "-m", "language_model", "preflight", "--work-dir", str(WORKING))

    corpus = WORKING / "data" / "raw" / "tiny-shakespeare.txt"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    event("download_corpus", "started", url=CORPUS_URL)
    with urllib.request.urlopen(CORPUS_URL, timeout=120) as response:
        content = response.read()
    corpus_hash = hashlib.sha256(content).hexdigest()
    if corpus_hash != CORPUS_SHA256:
        raise RuntimeError(f"corpus hash mismatch: {corpus_hash}")
    corpus.write_bytes(content)
    event("download_corpus", "completed", bytes=len(content), sha256=corpus_hash)

    config = PROJECT / "configs" / "kaggle_char_small.yaml"
    manifest = WORKING / "data" / "manifests" / "tiny_shakespeare" / "manifest.json"
    run(sys.executable, "-m", "language_model", "prepare-data", "--config", str(config), "--overwrite")
    run(sys.executable, "-m", "language_model", "inspect-data", "--manifest", str(manifest))
    run(sys.executable, "-m", "language_model", "train", "--config", str(config), "--manifest", str(manifest))
    event("kaggle_run", "completed", artifacts=str(WORKING / "artifacts"))


if __name__ == "__main__":
    main()
'''


def source_files() -> list[Path]:
    files: list[Path] = []
    for relative in INCLUDE_ROOTS:
        path = ROOT / relative
        if path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file() and "__pycache__" not in item.parts)
        elif path.is_file():
            files.append(path)
    return sorted(set(files), key=lambda item: item.relative_to(ROOT).as_posix())


def build_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in source_files():
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    archive = build_archive()
    digest = hashlib.sha256(archive).hexdigest()
    runner = RUNNER_TEMPLATE.replace("__SNAPSHOT_SHA256__", digest).replace(
        "__SNAPSHOT_BASE64__", base64.b64encode(archive).decode("ascii")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(runner, encoding="utf-8")
    summary = {"output": str(args.output), "snapshot_bytes": len(archive), "snapshot_sha256": digest}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
