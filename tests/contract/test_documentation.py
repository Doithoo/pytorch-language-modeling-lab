from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def test_local_markdown_links_resolve() -> None:
    failures: list[str] = []
    for document in sorted(ROOT.rglob("*.md")):
        if any(part in {".venv", "dist", "build"} or part.endswith(".egg-info") for part in document.parts):
            continue
        for raw_target in LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / unquote(target)).resolve()
            if not resolved.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {raw_target}")
    assert not failures, "broken local links:\n" + "\n".join(failures)
