.PHONY: sync lint format-check typecheck test build check kaggle-runner

sync:
	uv sync --locked --extra dev

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest -W error::DeprecationWarning

build:
	uv build

kaggle-runner:
	uv run python scripts/kaggle/build_snapshot.py

check: lint format-check typecheck test build
