.PHONY: setup lint format test db-up db-down etl split train-lgbm bench-mlp train-mlp cross-day compare pipeline demo

setup:            ## install deps and git hooks
	uv sync
	@uv run python -c "import lightgbm" 2>/dev/null || { \
	  echo "LightGBM cannot load OpenMP (libomp). On macOS run: scripts/fix_libomp_macos.sh"; \
	  echo "(or install native Homebrew and: brew install libomp). See README > Troubleshooting."; \
	  exit 1; }
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

db-up:
	docker compose up -d db

db-down:
	docker compose down

etl:              ## Day 2: data/raw/**/*.csv -> data/processed/flows.parquet + docs/data_card.md
	uv run ae etl

split:            ## Day 3: stratified 70/15/15 + cross-day split
	uv run ae split

train-lgbm:       ## Day 3: LightGBM baseline -> models/lgbm + reports/lgbm_metrics.json
	uv run ae train-lgbm

pipeline: etl split train-lgbm train-mlp cross-day compare

demo:             ## whole pipeline on small synthetic data in data/demo (no download needed)
	uv run ae demo

bench-mlp:
	uv run ae bench-mlp

train-mlp:
	uv run ae train-mlp

cross-day:
	uv run ae cross-day

compare:
	uv run ae compare
