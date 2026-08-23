"""Compare completed run summaries and report only valid metric rankings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _rank(rows: list[dict[str, Any]], metric: str) -> list[str]:
    available = [row for row in rows if row.get(metric) is not None]
    return [row["run"] for row in sorted(available, key=lambda row: float(row[metric]))]


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
                "manifest_identity": summary.get("manifest_identity"),
                "tokenizer_fingerprint": summary.get("tokenizer_fingerprint"),
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

    identities = {(row["manifest_identity"], row["tokenizer_fingerprint"]) for row in rows}
    same_evaluation_contract = len(identities) == 1 and all(
        row["manifest_identity"] and row["tokenizer_fingerprint"] for row in rows
    )
    result: dict[str, Any] = {
        "schema_version": 2,
        "runs": rows,
        "comparability": {
            "test_perplexity": {
                "comparable": same_evaluation_contract,
                "reason": (
                    "same manifest and tokenizer"
                    if same_evaluation_contract
                    else "requires the same manifest_identity and tokenizer_fingerprint"
                ),
            },
            "text_normalized_metrics": {
                "comparable": all(row["test_bits_per_character"] is not None for row in rows),
                "reason": "bits-per-character/byte are the cross-tokenizer metrics",
            },
        },
        "ranking_by_test_perplexity": _rank(rows, "test_perplexity") if same_evaluation_contract else [],
        "ranking_by_test_bits_per_character": _rank(rows, "test_bits_per_character"),
    }
    return result
