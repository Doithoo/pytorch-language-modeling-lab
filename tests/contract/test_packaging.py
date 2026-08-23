from __future__ import annotations

import ast
import base64
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

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
    module = ast.parse(source)
    payload = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "SNAPSHOT_BASE64"
    )
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        assert not any(".egg-info/" in name or "__pycache__/" in name for name in archive.namelist())
    compile(source, str(output), "exec")


def test_snapshot_builder_embeds_selected_dataset_name(tmp_path: Path) -> None:
    output = tmp_path / "run_seed7.py"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle" / "build_snapshot.py"),
            "--config",
            str(ROOT / "configs" / "kaggle_char_small_sdpa_seed7.yaml"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    source = output.read_text(encoding="utf-8")
    assert 'CONFIG_NAME = "kaggle_char_small_sdpa_seed7.yaml"' in source
    assert 'DATASET_NAME = "tiny_shakespeare_seed7"' in source
