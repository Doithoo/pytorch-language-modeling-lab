from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tomllib

import language_model

ROOT = Path(__file__).resolve().parents[2]


def test_package_metadata_and_cli_entrypoint() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["scripts"]["lm"] == "language_model.cli:main"
    assert metadata["project"]["requires-python"] == ">=3.10,<3.13"
    assert language_model.__version__ == metadata["project"]["version"]


def test_kaggle_metadata_points_to_existing_runner() -> None:
    directory = ROOT / "docs" / "recorded-run" / "kaggle"
    metadata = json.loads((directory / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == "true"
    assert (directory / metadata["code_file"]).is_file()


def test_snapshot_builder_creates_compilable_runner(tmp_path: Path) -> None:
    output = tmp_path / "run_kaggle.py"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "kaggle" / "build_snapshot.py"), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    source = output.read_text(encoding="utf-8")
    assert "__SNAPSHOT_BASE64__" not in source
    compile(source, str(output), "exec")
