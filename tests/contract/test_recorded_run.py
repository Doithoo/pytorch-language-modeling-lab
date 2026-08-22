from __future__ import annotations

import ast
import base64
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDED = ROOT / "docs" / "recorded-run"


def test_recorded_run_evidence_is_internally_consistent() -> None:
    with (RECORDED / "metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    best = min(rows, key=lambda row: float(row["valid_loss"]))
    evaluation = json.loads((RECORDED / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    run = json.loads((RECORDED / "run.json").read_text(encoding="utf-8"))
    environment = json.loads((RECORDED / "environment.json").read_text(encoding="utf-8"))
    data = json.loads((RECORDED / "data-summary.json").read_text(encoding="utf-8"))

    assert int(best["step"]) == run["best_checkpoint"]["step"] == 3000
    assert float(best["valid_loss"]) == evaluation["valid"]["loss"]
    assert evaluation["manifest_identity"] == run["manifest_identity"] == data["manifest_identity"]
    assert evaluation["checkpoint_sha256"] == run["best_checkpoint"]["sha256"]
    assert environment["cuda"]["name"] == "Tesla T4"
    assert environment["cuda"]["compatible"] is True
    frozen_source = RECORDED / "kernel-v2" / "run_kaggle.py"
    module = ast.parse(frozen_source.read_text(encoding="utf-8"))
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"SNAPSHOT_SHA256", "SNAPSHOT_BASE64"}
    }
    archive = base64.b64decode(values["SNAPSHOT_BASE64"])
    assert values["SNAPSHOT_SHA256"] == run["source_snapshot"]["sha256"] == hashlib.sha256(archive).hexdigest()
    assert len(archive) == run["source_snapshot"]["bytes"]
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert "pyproject.toml" in bundle.namelist()
    assert (RECORDED / "training-curves.png").stat().st_size > 10_000
