from __future__ import annotations

import csv
import json
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
    assert (RECORDED / "training-curves.png").stat().st_size > 10_000
