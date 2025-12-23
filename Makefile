.PHONY: help install dev clean test lint format typecheck docs build publish

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode
	pip install -e .

dev: ## Install package with development dependencies
	pip install -e ".[dev,all]"

clean: ## Remove build artifacts and cache files
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=rag_toolkit --cov-report=html --cov-report=term

lint: ## Run linting (ruff)
	ruff check src/rag_toolkit tests

format: ## Format code (black + isort)
	black src/rag_toolkit tests examples scripts
	isort src/rag_toolkit tests examples scripts

format-check: ## Check code formatting without changes
	black --check src/rag_toolkit tests examples scripts
	isort --check-only src/rag_toolkit tests examples scripts

typecheck: ## Run type checking (mypy)
	mypy src/rag_toolkit

check: format-check lint typecheck test ## Run all checks

docs: ## Build documentation
	cd docs && make html

docs-serve: docs ## Build and serve documentation
	cd docs/build/html && python -m http.server 8000

build: clean ## Build distribution packages
	python -m build

publish-test: build ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	twine upload dist/*

fix-imports: ## Fix src. imports to rag_toolkit.
	python scripts/fix_imports.py --root=src/rag-toolkit

# Docker targets
docker-up: ## Start all services (Milvus, Qdrant, Ollama)
	./docker/docker.sh up all

docker-up-milvus: ## Start Milvus only
	./docker/docker.sh up milvus

docker-up-qdrant: ## Start Qdrant only
	./docker/docker.sh up qdrant

docker-down: ## Stop all services
	./docker/docker.sh down all

docker-down-milvus: ## Stop Milvus
	./docker/docker.sh down milvus

docker-down-qdrant: ## Stop Qdrant
	./docker/docker.sh down qdrant

docker-restart: ## Restart all services
	./docker/docker.sh restart all

docker-logs: ## View logs from all services
	./docker/docker.sh logs all

docker-ps: ## Show running services
	./docker/docker.sh ps

docker-health: ## Check health of all services
	./docker/docker.sh health

docker-clean: ## Stop and remove all data (dangerous!)
	./docker/docker.sh clean all

docker-pull-models: ## Pull Ollama models
	./docker/docker.sh pull-models

# Development workflows
dev-setup: dev docker-up docker-pull-models ## Complete development setup
	@echo ""
	@echo "✅ Development environment ready!"
	@echo "   - Milvus:  http://localhost:19530 (UI: http://localhost:9091)"
	@echo "   - Qdrant:  http://localhost:6333 (Dashboard: http://localhost:6333/dashboard)"
	@echo "   - Ollama:  http://localhost:11434"
	@echo ""
	@echo "Next steps:"
	@echo "   1. Run tests: make test"
	@echo "   2. Try examples: python examples/quickstart.py"
	@echo "   3. Check docs: make docs-serve"

dev-teardown: docker-down ## Stop all development services
	@echo "✅ Development services stopped"

# Integration tests with Docker
test-integration: docker-up ## Run integration tests
	@echo "Waiting for services to be ready..."
	@sleep 10
	pytest tests/integration -v -m integration || true
	$(MAKE) docker-down
