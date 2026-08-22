"""Compare completed run summaries without loading model checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_runs(run_directories: list[str | Path]) -> dict[str, Any]:
    if len(run_directories) < 2:
        raise ValueError("compare-runs requires at least two run directories")
    rows: list[dict[str, Any]] = []
    for raw_directory in run_directories:
        directory = Path(raw_directory)
        summary_path = directory / "run.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"run summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise ValueError(f"run is not completed: {directory}")
        valid = summary.get("valid", {})
        test = summary.get("test", {})
        rows.append(
            {
                "run": str(directory),
                "parameters": summary.get("parameters"),
                "global_step": summary.get("global_step"),
                "processed_tokens": summary.get("processed_tokens"),
                "valid_loss": valid.get("loss"),
                "valid_perplexity": valid.get("perplexity"),
                "test_loss": test.get("loss"),
                "test_perplexity": test.get("perplexity"),
                "test_bits_per_character": test.get("bits_per_character"),
                "test_bits_per_byte": test.get("bits_per_byte"),
                "test_accuracy_top1": test.get("accuracy_top1"),
                "tokens_per_second": summary.get("last_metrics", {}).get("tokens_per_second"),
                "peak_memory_bytes": summary.get("peak_memory_bytes"),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: float(row["test_perplexity"]) if row["test_perplexity"] is not None else float("inf"),
    )
    return {"runs": rows, "ranking_by_test_perplexity": [row["run"] for row in ranked]}
