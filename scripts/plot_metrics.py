"""Plot train/validation metrics from a completed run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, default=Path("metrics.png"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    with args.metrics.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metrics file has no data rows")
    steps = [int(row["step"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    valid_loss = [float(row["valid_loss"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(steps, train_loss, label="train loss")
    axis.plot(steps, valid_loss, label="valid loss")
    axis.set(xlabel="optimizer step", ylabel="token cross-entropy", title="Language-model training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
