"""Prepare normalized, tokenized, versioned corpus files."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExperimentConfig
from ..tokenization import LanguageModelTokenizer, train_tokenizer
from ..utils import sha256_file, write_json
from .clean import clean_documents
from .manifest import MANIFEST_SCHEMA_VERSION, default_manifest_path, manifest_identity
from .sources import Document, read_documents
from .split import split_documents


def _window_count(token_count: int, stride: int) -> int:
    if token_count < 2:
        return 0
    return ((token_count - 2) // stride) + 1


def _encode_split(
    documents: list[Document],
    tokenizer: LanguageModelTokenizer,
    output_path: Path,
    stride: int,
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    offsets: list[dict[str, Any]] = []
    token_count = 0
    window_count = 0
    character_count = 0
    utf8_byte_count = 0
    with output_path.open("wb") as handle:
        for document in documents:
            ids = tokenizer.encode(document.text, add_bos=True, add_eos=True)
            if len(ids) < 2:
                continue
            start = token_count
            np.asarray(ids, dtype=np.uint32).tofile(handle)
            token_count += len(ids)
            windows = _window_count(len(ids), stride)
            characters = len(document.text)
            utf8_bytes = len(document.text.encode("utf-8"))
            window_count += windows
            character_count += characters
            utf8_byte_count += utf8_bytes
            offsets.append(
                {
                    "document_id": document.document_id,
                    "source": document.source,
                    "start": start,
                    "end": token_count,
                    "tokens": len(ids),
                    "windows": windows,
                    "characters": characters,
                    "utf8_bytes": utf8_bytes,
                }
            )
    if not offsets or window_count == 0:
        raise ValueError("a prepared split contains no usable next-token windows")
    return offsets, token_count, window_count, character_count, utf8_byte_count


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish_pair(data_temp: Path, data_target: Path, manifest_temp: Path, manifest_target: Path) -> None:
    targets = ((data_temp, data_target), (manifest_temp, manifest_target))
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for _, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
                os.replace(target, backup)
                backups.append((target, backup))
        for temporary, target in targets:
            os.replace(temporary, target)
            published.append(target)
    except BaseException:
        for target in reversed(published):
            _remove(target)
        for target, backup in reversed(backups):
            os.replace(backup, target)
        raise
    else:
        for _, backup in backups:
            _remove(backup)


def prepare_data(cfg: ExperimentConfig, *, overwrite: bool = False) -> Path:
    documents, source_files = read_documents(cfg.data.corpus_dir)
    cleaned, cleaning_stats = clean_documents(documents)
    splits = split_documents(
        cleaned,
        cfg.data.split_ratios,
        cfg.data.seed,
        strategy=cfg.data.split_strategy,
        group_by_source=cfg.data.group_by_source,
    )
    tokenizer = train_tokenizer((item.text for item in splits["train"]), cfg.tokenizer)

    data_target = cfg.data.processed_dir / cfg.data.dataset_name
    manifest_target = cfg.data.manifest_dir / cfg.data.dataset_name
    if not overwrite and (data_target.exists() or manifest_target.exists()):
        existing = data_target if data_target.exists() else manifest_target
        raise FileExistsError(f"target already exists; pass --overwrite to replace it: {existing}")

    data_parent = cfg.data.processed_dir.resolve().parent
    manifest_parent = cfg.data.manifest_dir.resolve().parent
    data_parent.mkdir(parents=True, exist_ok=True)
    manifest_parent.mkdir(parents=True, exist_ok=True)
    data_temp = Path(tempfile.mkdtemp(prefix=f".{cfg.data.dataset_name}-", dir=data_parent))
    manifest_temp = Path(tempfile.mkdtemp(prefix=f".{cfg.data.dataset_name}-", dir=manifest_parent))
    try:
        split_metadata: dict[str, Any] = {}
        for name, split_documents_list in splits.items():
            file_name = f"{name}.bin"
            offsets, tokens, windows, characters, utf8_bytes = _encode_split(
                split_documents_list,
                tokenizer,
                data_temp / file_name,
                cfg.data.stride,
            )
            split_metadata[name] = {
                "file": file_name,
                "sha256": sha256_file(data_temp / file_name),
                "documents": len(offsets),
                "tokens": tokens,
                "windows": windows,
                "characters": characters,
                "utf8_bytes": utf8_bytes,
                "offsets": offsets,
            }

        tokenizer.save(manifest_temp / "tokenizer")
        source_metadata = [
            {
                "path": path.relative_to(cfg.data.corpus_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_files
        ]
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset_name": cfg.data.dataset_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(cfg.data.corpus_dir.resolve()),
            "source_files": source_metadata,
            "normalization": "NFKC+line-endings+horizontal-space-v1",
            "cleaning": cleaning_stats,
            "split_ratios": cfg.data.split_ratios,
            "split_seed": cfg.data.seed,
            "split_strategy": cfg.data.split_strategy,
            "group_by_source": cfg.data.group_by_source,
            "window_mode": cfg.data.window_mode,
            "context_length": cfg.data.context_length,
            "stride": cfg.data.stride,
            "dtype": "uint32",
            "processed_root": os.path.relpath(data_target.resolve(), manifest_target.resolve()),
            "processed_root_mode": "manifest_relative",
            "tokenizer": {
                "name": cfg.tokenizer.name,
                "vocab_size": tokenizer.vocab_size,
                "fingerprint": tokenizer.fingerprint(),
                "special_token_ids": {
                    "pad": tokenizer.pad_id,
                    "bos": tokenizer.bos_id,
                    "eos": tokenizer.eos_id,
                    "unk": tokenizer.unk_id,
                },
            },
            "splits": split_metadata,
        }
        manifest["identity"] = manifest_identity(manifest)
        write_json(manifest_temp / "manifest.json", manifest)
        write_json(
            manifest_temp / "summary.json",
            {
                "dataset_name": cfg.data.dataset_name,
                "identity": manifest["identity"],
                "vocab_size": tokenizer.vocab_size,
                "splits": {
                    name: {
                        key: value
                        for key, value in metadata.items()
                        if key in {"documents", "tokens", "windows", "characters", "utf8_bytes"}
                    }
                    for name, metadata in split_metadata.items()
                },
            },
        )
        _publish_pair(data_temp, data_target, manifest_temp, manifest_target)
    except BaseException:
        shutil.rmtree(data_temp, ignore_errors=True)
        shutil.rmtree(manifest_temp, ignore_errors=True)
        raise
    return default_manifest_path(cfg.data.manifest_dir, cfg.data.dataset_name)


def describe_manifest(path: str | Path) -> str:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return json.dumps(
        {
            "dataset_name": manifest["dataset_name"],
            "identity": manifest["identity"],
            "tokenizer": manifest["tokenizer"],
            "splits": {
                name: {key: value for key, value in split.items() if key in {"documents", "tokens", "windows"}}
                for name, split in manifest["splits"].items()
            },
        },
        ensure_ascii=False,
        indent=2,
    )
