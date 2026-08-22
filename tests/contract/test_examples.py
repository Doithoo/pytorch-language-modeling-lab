from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_examples_execute() -> None:
    for name in ("01_bigram_forward.py", "02_causal_attention.py"):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "examples" / name)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip()
