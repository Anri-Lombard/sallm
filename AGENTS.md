# Repository Guidelines

## Project Structure & Module Organization
- `src/main/sallm` hosts the core training pipelines and utilities; treat `src/main` as the Python package root.
- Configuration YAML files live in `src/conf` (datasets, tokenizers, finetune, templates, eval). Group new configs with their peers.
- `scripts` contains CLI helpers; mirror module naming when adding automation.
- `tokenizer` stores tokenizer training assets; keep raw or intermediate datasets in `data`.
- Place tests under `tests`, mirroring package paths (`tests/main/test_<module>.py`) for quick discovery.

## Environment & Tooling
- Always run `conda activate base` before project commands so Poetry resolves the correct environment.
- Install dependencies with `poetry install`; only enter `poetry shell` if you understand its isolation model.

## Build, Test, and Development Commands
- `make format` runs Black, Ruff format, and Ruff autofix.
- `make lint` executes Ruff linting and YAML linting for configuration files.
- `make type` enforces mypy across `src` and `tokenizer`.
- `make test` runs `pytest -q`; append `-k <pattern>` to target specific suites.
- `make check` chains the full formatter, lint, type, test, and pre-commit sweep.

## Coding Style & Naming Conventions
- Python only: no inline comments or docstrings in source files; rely on expressive, type-hinted signatures instead.
- Use four-space indentation, 88-character lines, and let Ruff/Black dictate formatting.
- Keep imports at the top of modules and avoid try/except wrappers around them.
- Adopt snake_case for functions and variables, UpperCamelCase for classes, and align test names with their feature area.

## Testing Guidelines
- Use pytest with files named `test_*.py` and classes prefixed `Test`; place fixtures alongside their tests.
- Target deterministic coverage around configuration parsing, data pipelines, and training/evaluation utilities.
- Avoid slow or network-bound tests unless guarded by explicit markers.

## Commit & Pull Request Guidelines
- Follow the repository history: concise, imperative commit headers (e.g., `Fix eval yamls`, `Add tokenizer metrics`).
- Keep changes scoped; run `make check` locally before pushing.
- Pull requests need a focused summary, validation evidence, and linked issues or context. Attach metrics or screenshots for training/eval updates when relevant.
