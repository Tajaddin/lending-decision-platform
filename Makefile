.PHONY: install lint format typecheck test test-integration test-bats alerts-test check secret-scan download pipeline \
        train model-card score-batch bench ablation up down replay load-test tf-validate bundle-validate backlog

YEARS ?= 2023,2024,2025
RAW ?= data/raw
RATE ?= 50
KEY ?= local-dev-key-replay-000002

install:            ## all extras: Spark, TensorFlow, dev tools
	uv sync --all-extras

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check --fix .

typecheck:
	uv run mypy

test:               ## unit + Spark + TensorFlow tests with coverage (Java 17+ needed for Spark)
	uv run pytest -m "not integration" --cov=ldp --cov-report=term-missing --cov-report=xml

test-integration:   ## real TF Serving, Pub/Sub and BigQuery emulators via Testcontainers (Docker)
	uv run pytest -m integration -v

test-bats:
	shellcheck scripts/*.sh && bats tests/bats

alerts-test:
	docker run --rm -v "$(CURDIR)/observability:/w" -w /w --entrypoint promtool prom/prometheus:v3.5.0 test rules alerts_test.yml

check: lint typecheck test test-bats

secret-scan:        ## run before every push
	bash scripts/secret_scan.sh

download:           ## national HMDA snapshot files, ~2 GB zipped, 14 GB unzipped
	for y in $$(echo $(YEARS) | tr ',' ' '); do bash scripts/download_hmda.sh $$y $(RAW); done

pipeline:           ## bronze -> silver -> quality -> gold -> leakage audit -> training export
	uv run ldp pipeline --years $(YEARS) --raw-dir $(RAW)

train:              ## TensorFlow model + baselines, evaluation, fairness, export to models/ldp/1
	uv run ldp train --version 1 && uv run ldp model-card --version 1

model-card:
	uv run ldp model-card --version 1

score-batch:        ## score 2025 in Spark with the SavedModel
	uv run ldp score-batch --year 2025 --version 1

bench:              ## silver stage at 1/2/4/8/16 cores on 2025 (after `make pipeline`)
	uv run ldp bench-silver --year 2025 --cores 1,2,4,8,16

ablation:           ## drop-one-input ablation and AIR per variant (less discriminatory alternative search)
	uv run python scripts/ablation.py

up:                 ## the whole online stack from local images
	docker compose up -d --build --wait --wait-timeout 300 api sink monitor prometheus grafana

down:
	docker compose down -v

replay:             ## 5,000 real 2025 applications through the API at RATE per second
	uv run ldp replay --url http://localhost:8200 --split data/sample/test --n 5000 --rate $(RATE) --api-key $(KEY)

load-test:          ## k6 at RATE requests per second for 60 s against the compose stack
	mkdir -p reports/k6 && docker run --rm --network ldp_default -v "$(CURDIR)/bench/k6:/scripts:ro" \
	  -v "$(CURDIR)/reports/k6:/out" -e BASE_URL=http://api:8080 -e API_KEY=$(KEY) -e RATE=$(RATE) \
	  grafana/k6:1.3.0 run --quiet /scripts/decisions.js

tf-validate:
	terraform -chdir=infra/terraform fmt -check -recursive && terraform -chdir=infra/terraform init -backend=false \
	  -input=false && terraform -chdir=infra/terraform validate

bundle-validate:    ## Databricks bundle against the CLI's JSON schema (no workspace needed)
	cd databricks && databricks bundle schema > /tmp/bundle-schema.json && \
	  uv run check-jsonschema --schemafile /tmp/bundle-schema.json databricks.yml resources/*.yml

backlog:            ## create labels, milestones and issues on GitHub from docs/agile/backlog.json
	python scripts/agile/sync_backlog.py
