PY ?= python

.PHONY: format lint type test check export-req pre-commit-install pre-commit-check

format:
	poetry run black .
	poetry run ruff format .
	poetry run ruff check --fix .

lint:
	poetry run ruff check .
	@poetry run yamllint --config-file pyproject.toml environment.yml || \
		(poetry run pre-commit run yamllint --all-files)

type:
	poetry run mypy src tokenizer

test:
	poetry run pytest -q

pre-commit-check:
	poetry run pre-commit run --all-files || true

check: format lint type test pre-commit-check

export-req:
	poetry export -f requirements.txt --without-hashes -o requirements.txt

pre-commit-install:
	poetry run pre-commit install --install-hooks

deps-outdated:
	poetry show --outdated

deps-update:
	poetry update