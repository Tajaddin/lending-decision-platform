.PHONY: install lint format typecheck test test-bats check secret-scan download pipeline bench backlog

YEARS ?= 2023,2024,2025
RAW ?= data/raw

install:            ## all extras: Spark, TensorFlow, dev tools
	uv sync --all-extras

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check --fix .

typecheck:
	uv run mypy

test:               ## unit + Spark tests with coverage (Java 17+ needed for Spark)
	uv run pytest -m "not integration" --cov=ldp --cov-report=term-missing --cov-report=xml

test-bats:
	shellcheck scripts/*.sh && bats tests/bats

check: lint typecheck test test-bats

secret-scan:        ## run before every push
	bash scripts/secret_scan.sh

download:           ## national HMDA snapshot files, ~2 GB zipped, 14 GB unzipped
	for y in $$(echo $(YEARS) | tr ',' ' '); do bash scripts/download_hmda.sh $$y $(RAW); done

pipeline:           ## bronze -> silver -> quality -> gold -> leakage audit -> training export
	uv run ldp pipeline --years $(YEARS) --raw-dir $(RAW)

bench:              ## silver stage at 1/2/4/8/16 cores on 2025 (after `make pipeline`)
	uv run ldp bench-silver --year 2025 --cores 1,2,4,8,16

backlog:            ## create labels, milestones and issues on GitHub from docs/agile/backlog.json
	python scripts/agile/sync_backlog.py
