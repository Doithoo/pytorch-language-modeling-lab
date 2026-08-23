.PHONY: sync lint format-check typecheck test docs-check build package-check check kaggle-runner

sync:
	uv sync --locked --extra dev

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run python -m pytest $(shell find tests -type f -name 'test_*.py' | sort) -W error::DeprecationWarning

docs-check:
	uv run mkdocs build --strict

package-check: build
	uv run twine check dist/*

build:
	uv build

kaggle-runner:
	uv run python scripts/kaggle/build_snapshot.py

check: lint format-check typecheck test docs-check package-check
