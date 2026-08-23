from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
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


def test_core_documentation_has_bilingual_pairs() -> None:
    pairs = [
        ("docs/tutorial/README.md", "docs/tutorial/README.zh-CN.md"),
        ("docs/tutorial/00-basics.md", "docs/tutorial/00-basics.zh-CN.md"),
        ("docs/tutorial/01-environment.md", "docs/tutorial/01-environment.zh-CN.md"),
        ("docs/tutorial/02-data-and-tokenization.md", "docs/tutorial/02-data-and-tokenization.zh-CN.md"),
        ("docs/tutorial/03-models.md", "docs/tutorial/03-models.zh-CN.md"),
        ("docs/tutorial/04-training-and-resume.md", "docs/tutorial/04-training-and-resume.zh-CN.md"),
        ("docs/tutorial/05-evaluation-and-generation.md", "docs/tutorial/05-evaluation-and-generation.zh-CN.md"),
        ("docs/guides/first-run.md", "docs/guides/first-run.zh-CN.md"),
        ("docs/guides/using-your-data.md", "docs/guides/using-your-data.zh-CN.md"),
        ("docs/guides/experiments.md", "docs/guides/experiments.zh-CN.md"),
        ("docs/guides/adding-models.md", "docs/guides/adding-models.zh-CN.md"),
        ("docs/reference/cli-and-outputs.md", "docs/reference/cli-and-outputs.zh-CN.md"),
        ("docs/reference/dataset-card.md", "docs/reference/dataset-card.zh-CN.md"),
        ("docs/reference/model-card.md", "docs/reference/model-card.zh-CN.md"),
    ]
    missing = [pair for pair in pairs if any(not (ROOT / path).is_file() for path in pair)]
    assert not missing, f"missing bilingual documentation pairs: {missing}"
