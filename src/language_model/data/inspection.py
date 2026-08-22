"""Prepared-corpus inspection without model construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..tokenization import load_tokenizer
from ..utils import sha256_file
from .manifest import ManifestError, load_manifest


def inspect_prepared_data(
    manifest_path: str | Path,
    *,
    verify_hashes: bool = True,
    processed_root: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = load_manifest(path)
    tokenizer = load_tokenizer(path.parent / "tokenizer")
    expected_fingerprint = manifest["tokenizer"]["fingerprint"]
    if tokenizer.fingerprint() != expected_fingerprint:
        raise ManifestError("manifest and tokenizer fingerprints differ")
    if tokenizer.vocab_size != manifest["tokenizer"]["vocab_size"]:
        raise ManifestError("manifest and tokenizer vocabulary sizes differ")
    if processed_root is not None:
        root = Path(processed_root)
    else:
        root = Path(manifest["processed_root"])
        if not root.is_absolute():
            root = path.parent / root
    checks: dict[str, Any] = {}
    for split, metadata in manifest["splits"].items():
        data_path = root / metadata["file"]
        if not data_path.is_file():
            raise FileNotFoundError(f"prepared split missing: {data_path}")
        actual = sha256_file(data_path) if verify_hashes else None
        if verify_hashes and actual != metadata["sha256"]:
            raise ManifestError(f"prepared split hash mismatch: {split}")
        checks[split] = {
            "documents": metadata["documents"],
            "tokens": metadata["tokens"],
            "windows": metadata["windows"],
            "sha256": actual,
        }
    return {
        "dataset_name": manifest["dataset_name"],
        "identity": manifest["identity"],
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "splits": checks,
    }
