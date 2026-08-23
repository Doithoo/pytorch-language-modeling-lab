"""Versioned dataset manifest loading and identity validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..utils import sha256_json

MANIFEST_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """Raised when prepared data metadata is incomplete or inconsistent."""


def manifest_identity(manifest: dict[str, Any]) -> str:
    excluded = {"identity", "created_at", "source_root", "processed_root"}
    identity_input = {key: value for key, value in manifest.items() if key not in excluded}
    return sha256_json(identity_input)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"manifest field {field} must be an object")
    return value


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ManifestError(f"manifest field {field} must be a {qualifier} integer")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"manifest field {field} must be a non-empty string")
    return value


def _hash(value: Any, field: str) -> str:
    text = _string(value, field)
    if _SHA256.fullmatch(text) is None:
        raise ManifestError(f"manifest field {field} must be a lowercase SHA-256")
    return text


def _validate_manifest(manifest: dict[str, Any]) -> None:
    _string(manifest.get("dataset_name"), "dataset_name")
    _string(manifest.get("processed_root"), "processed_root")
    _positive_int(manifest.get("context_length"), "context_length")
    _positive_int(manifest.get("stride"), "stride")
    if manifest.get("dtype") != "uint32":
        raise ManifestError("manifest field dtype must be uint32")

    tokenizer = _mapping(manifest.get("tokenizer"), "tokenizer")
    vocab_size = _positive_int(tokenizer.get("vocab_size"), "tokenizer.vocab_size")
    _string(tokenizer.get("name"), "tokenizer.name")
    _hash(tokenizer.get("fingerprint"), "tokenizer.fingerprint")
    special_ids = _mapping(tokenizer.get("special_token_ids"), "tokenizer.special_token_ids")
    ids = [
        _positive_int(special_ids.get(name), f"tokenizer.special_token_ids.{name}", allow_zero=True)
        for name in ("pad", "bos", "eos", "unk")
    ]
    if len(set(ids)) != 4 or any(token_id >= vocab_size for token_id in ids):
        raise ManifestError("manifest special token ids must be unique and within the vocabulary")

    source = _mapping(manifest.get("source", {}), "source")
    for field in ("url", "version", "license"):
        value = source.get(field)
        if value is not None:
            _string(value, f"source.{field}")

    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ManifestError("manifest field source_files must be a non-empty list")
    for index, raw_source in enumerate(source_files):
        source = _mapping(raw_source, f"source_files[{index}]")
        _string(source.get("path"), f"source_files[{index}].path")
        _positive_int(source.get("bytes"), f"source_files[{index}].bytes", allow_zero=True)
        _hash(source.get("sha256"), f"source_files[{index}].sha256")

    splits = _mapping(manifest.get("splits"), "splits")
    for name in ("train", "valid", "test"):
        split = _mapping(splits.get(name), f"splits.{name}")
        _string(split.get("file"), f"splits.{name}.file")
        _hash(split.get("sha256"), f"splits.{name}.sha256")
        documents = _positive_int(split.get("documents"), f"splits.{name}.documents")
        tokens = _positive_int(split.get("tokens"), f"splits.{name}.tokens")
        windows = _positive_int(split.get("windows"), f"splits.{name}.windows")
        offsets = split.get("offsets")
        if not isinstance(offsets, list) or len(offsets) != documents:
            raise ManifestError(f"manifest offsets do not match document count for split {name}")
        expected_start = 0
        counted_windows = 0
        for index, raw_offset in enumerate(offsets):
            offset = _mapping(raw_offset, f"splits.{name}.offsets[{index}]")
            start = _positive_int(offset.get("start"), f"splits.{name}.offsets[{index}].start", allow_zero=True)
            end = _positive_int(offset.get("end"), f"splits.{name}.offsets[{index}].end")
            item_tokens = _positive_int(offset.get("tokens"), f"splits.{name}.offsets[{index}].tokens")
            item_windows = _positive_int(offset.get("windows"), f"splits.{name}.offsets[{index}].windows")
            _string(offset.get("document_id"), f"splits.{name}.offsets[{index}].document_id")
            _string(offset.get("source"), f"splits.{name}.offsets[{index}].source")
            if start != expected_start or end - start != item_tokens:
                raise ManifestError(f"manifest offsets are not contiguous for split {name}")
            expected_start = end
            counted_windows += item_windows
        if expected_start != tokens or counted_windows != windows:
            raise ManifestError(f"manifest token/window totals are inconsistent for split {name}")


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid manifest JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("manifest root must be an object")
    manifest: dict[str, Any] = raw
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"unsupported manifest schema: {manifest.get('schema_version')!r}")
    _validate_manifest(manifest)
    expected = manifest_identity(manifest)
    if manifest.get("identity") != expected:
        raise ManifestError("manifest identity does not match its contents")
    return manifest


def default_manifest_path(manifest_dir: str | Path, dataset_name: str) -> Path:
    return Path(manifest_dir) / dataset_name / "manifest.json"
