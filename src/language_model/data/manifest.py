"""Versioned dataset manifest loading and identity validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils import sha256_json

MANIFEST_SCHEMA_VERSION = 1


class ManifestError(ValueError):
    """Raised when prepared data metadata is incomplete or inconsistent."""


def manifest_identity(manifest: dict[str, Any]) -> str:
    excluded = {"identity", "created_at", "source_root", "processed_root"}
    identity_input = {key: value for key, value in manifest.items() if key not in excluded}
    return sha256_json(identity_input)


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"unsupported manifest schema: {manifest.get('schema_version')!r}")
    expected = manifest_identity(manifest)
    if manifest.get("identity") != expected:
        raise ManifestError("manifest identity does not match its contents")
    for split in ("train", "valid", "test"):
        if split not in manifest.get("splits", {}):
            raise ManifestError(f"manifest is missing split: {split}")
    return manifest


def default_manifest_path(manifest_dir: str | Path, dataset_name: str) -> Path:
    return Path(manifest_dir) / dataset_name / "manifest.json"
