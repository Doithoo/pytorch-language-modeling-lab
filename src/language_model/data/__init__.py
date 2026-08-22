"""Corpus preparation and memory-mapped next-token datasets."""

from .dataset import TokenWindowDataset
from .inspection import inspect_prepared_data
from .manifest import ManifestError, default_manifest_path, load_manifest
from .prepare import prepare_data
from .sources import CorpusError, Document, read_documents

__all__ = [
    "CorpusError",
    "Document",
    "ManifestError",
    "TokenWindowDataset",
    "default_manifest_path",
    "inspect_prepared_data",
    "load_manifest",
    "prepare_data",
    "read_documents",
]
