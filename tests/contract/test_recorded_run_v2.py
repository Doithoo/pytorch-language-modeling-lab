from __future__ import annotations

import ast
import base64
import csv
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDED = ROOT / "docs" / "recorded-run" / "v2"


def test_v2_recorded_run_provenance_and_metrics() -> None:
    provenance = json.loads((RECORDED / "provenance.json").read_text(encoding="utf-8"))
    run = json.loads((RECORDED / "run.json").read_text(encoding="utf-8"))
    evaluation = json.loads((RECORDED / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((RECORDED / "metrics.csv").open(encoding="utf-8")))

    assert provenance["release_version"] == "0.2.0"
    assert provenance["kernel_version"] == 3
    assert run["status"] == "completed"
    assert len(rows) == 20
    assert run["best_checkpoint"]["step"] == 2750
    assert evaluation["checkpoint_sha256"] == provenance["checkpoints"]["best"]["sha256"]
    assert evaluation["manifest_identity"] == provenance["manifest_identity"]
    assert evaluation["test"]["accuracy_top5"] > evaluation["test"]["accuracy_top1"]

    runner = ast.parse((ROOT / "docs/recorded-run/v2/runner-v3.py").read_text(encoding="utf-8"))
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in runner.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"SNAPSHOT_SHA256", "SNAPSHOT_BASE64"}
    }
    archive = base64.b64decode(values["SNAPSHOT_BASE64"])
    assert values["SNAPSHOT_SHA256"] == provenance["source_snapshot"]["sha256"]
    assert len(archive) == provenance["source_snapshot"]["bytes"]
    assert hashlib.sha256(archive).hexdigest() == values["SNAPSHOT_SHA256"]
    with zipfile.ZipFile(BytesIO(archive)) as bundle:
        assert "configs/kaggle_char_small_sdpa.yaml" in bundle.namelist()
        assert "requirements.lock" in bundle.namelist()


def test_v2_rope_recorded_run_provenance() -> None:
    directory = RECORDED / "rope"
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    evaluation = json.loads((directory / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    assert provenance["kernel_version"] == 1
    assert provenance["positional_encoding"] == "rotary"
    assert run["status"] == "completed"
    assert run["manifest_identity"] == provenance["manifest_identity"]
    assert evaluation["checkpoint_sha256"] == provenance["checkpoints"]["best"]["sha256"]
    assert evaluation["test"]["perplexity"] < 3.6
    assert evaluation["test"]["accuracy_top1"] > 0.60


def test_multiseed_rope_replication_is_paired_and_consistent() -> None:
    summary = json.loads((RECORDED / "multiseed" / "summary.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == [42, 7, 123]
    assert len(summary["runs"]) == 3
    for pair in summary["runs"]:
        assert pair["rope_sdpa"]["test_perplexity"] < pair["learned_sdpa"]["test_perplexity"]
        assert pair["rope_sdpa"]["test_accuracy_top1"] > pair["learned_sdpa"]["test_accuracy_top1"]
        if pair["seed"] in {7, 123}:
            seed_dir = RECORDED / "multiseed" / f"seed{pair['seed']}"
            learned = json.loads((seed_dir / "sdpa" / "run.json").read_text(encoding="utf-8"))
            rope = json.loads((seed_dir / "rope" / "run.json").read_text(encoding="utf-8"))
            assert learned["manifest_identity"] == rope["manifest_identity"] == pair["manifest_identity"]
            assert learned["test"]["perplexity"] == pair["learned_sdpa"]["test_perplexity"]
            assert rope["test"]["perplexity"] == pair["rope_sdpa"]["test_perplexity"]
    stats = summary["paired_test_perplexity"]
    assert stats["rope_sdpa_mean"] < stats["learned_sdpa_mean"]
    assert stats["rope_minus_learned_mean"] < 0
