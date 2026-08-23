# Package Architecture

The package is intentionally organized by domain boundary:

1. `config.py`, `preflight.py`, `data/`, and `tokenization/`
2. `models/`
3. `training/`
4. `evaluation/` and `generation/`
5. `inference/` and `cli.py`

The CLI composes these modules and remains thin. Dataset parsing, model construction, and checkpoint validation stay in their owning modules.

When a public contract changes, update the matching reference page, tests, and both language versions of the documentation.
