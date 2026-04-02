.PHONY: install install-dev test lint format clean docker-build docker-up docker-down run-api

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=av_sim_arena --cov-report=term-missing

lint:
	ruff check av_sim_arena/ tests/

format:
	ruff format av_sim_arena/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f leaderboard.db

run-api:
	uvicorn av_sim_arena.leaderboard.api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
