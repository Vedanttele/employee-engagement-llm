.PHONY: install install-dev test lint format run-api run-app docker-build docker-up docker-down generate

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	python -m spacy download en_core_web_lg

test:
	pytest tests/ -v --cov=employee_engagement --cov-report=term-missing

lint:
	ruff check employee_engagement/ tests/

format:
	ruff format employee_engagement/ tests/

run-api:
	uvicorn employee_engagement.api.main:app --reload --host 0.0.0.0 --port 8000

run-app:
	streamlit run app/streamlit_app.py --server.port 8501

generate:
	python -m employee_engagement.data.generator

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
