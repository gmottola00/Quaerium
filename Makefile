.PHONY: help install dev clean test lint format typecheck docs build publish

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode
	uv pip install -e .

dev: ## Install package with development dependencies
	uv pip install -e ".[dev,all]"

clean: ## Remove build artifacts and cache files
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=rag_toolkit --cov-report=html --cov-report=term

test-unified: ## Run unified vector store tests
	uv run pytest tests/test_infra/test_vectorstores/test_unified.py -v

test-vectorstores: ## Run all vector store tests
	uv run pytest tests/test_infra/test_vectorstores/ -v

test-milvus: ## Run Milvus tests only
	uv run pytest tests/test_infra/test_vectorstores/ -k milvus -v

test-qdrant: ## Run Qdrant tests only
	uv run pytest tests/test_infra/test_vectorstores/ -k qdrant -v

test-chroma: ## Run ChromaDB tests only
	uv run pytest tests/test_infra/test_vectorstores/ -k chroma -v

test-metadata: ## Run metadata extraction tests
	uv run pytest tests/test_core/metadata/ -v

test-enrichment: ## Run metadata enrichment tests
	uv run pytest tests/test_chunking/test_enrichment.py -v

test-metadata-all: ## Run all metadata and enrichment tests with coverage
	uv run pytest tests/test_core/metadata/ tests/test_chunking/test_enrichment.py -v --cov=rag_toolkit.core.metadata --cov=rag_toolkit.core.chunking.enrichment --cov-report=term --cov-report=html

test-metadata-quick: ## Quick test of metadata extraction and enrichment (no coverage)
	uv run pytest tests/test_core/metadata/ tests/test_chunking/test_enrichment.py -v --tb=short

test-graph: ## Run graph store unit tests (no Neo4j required)
	uv run pytest tests/test_core/test_graphstore_protocol.py -v

test-graph-integration: ## Run graph store integration tests (requires Neo4j)
	uv run pytest tests/integration/test_neo4j_service.py -v

test-graph-all: ## Run all graph store tests with coverage
	uv run pytest tests/test_core/test_graphstore_protocol.py tests/integration/test_neo4j_service.py -v --cov=rag_toolkit.core.graphstore --cov=rag_toolkit.infra.graphstores --cov-report=term --cov-report=html

benchmark: ## Run performance benchmarks
	uv run pytest tests/benchmarks/ --benchmark-only --benchmark-autosave

benchmark-compare: ## Compare current benchmarks with last run
	uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare

benchmark-report: ## Generate HTML report from benchmarks
	@uv run python scripts/generate_benchmark_report.py

benchmark-docs: ## Update benchmark report in documentation
	@bash scripts/update_benchmark_docs.sh

benchmark-clean: ## Remove benchmark results
	rm -rf .benchmarks/ benchmark_report.html

lint: ## Run linting (ruff)
	uv run ruff check src/rag_toolkit tests

format: ## Format code (black + isort)
	uv run black src/rag_toolkit tests examples scripts
	uv run isort src/rag_toolkit tests examples scripts

format-check: ## Check code formatting without changes
	uv run black --check src/rag_toolkit tests examples scripts
	uv run isort --check-only src/rag_toolkit tests examples scripts

typecheck: ## Run type checking (mypy)
	uv run mypy src/rag_toolkit

check: format-check lint typecheck test ## Run all checks

docs: ## Build MkDocs documentation
	uv run mkdocs build

docs-serve: ## Serve MkDocs documentation locally (http://localhost:8000)
	uv run mkdocs serve

docs-build-strict: ## Build documentation in strict mode (fails on warnings)
	uv run mkdocs build --strict

docs-clean: ## Clean built documentation
	rm -rf site/

build: clean ## Build distribution packages
	python -m build

publish-test: build ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	twine upload dist/*

fix-imports: ## Fix src. imports to rag_toolkit.
	python scripts/fix_imports.py --root=src/rag-toolkit

# Docker targets
docker-up: ## Start all services (Milvus, Qdrant, ChromaDB, Ollama)
	./docker/docker.sh up all

docker-up-milvus: ## Start Milvus only
	./docker/docker.sh up milvus

docker-up-qdrant: ## Start Qdrant only
	./docker/docker.sh up qdrant

docker-up-chroma: ## Start ChromaDB only
	./docker/docker.sh up chroma

docker-up-neo4j: ## Start Neo4j only
	./docker/docker.sh up neo4j

docker-down: ## Stop all services
	./docker/docker.sh down all

docker-down-milvus: ## Stop Milvus
	./docker/docker.sh down milvus

docker-down-qdrant: ## Stop Qdrant
	./docker/docker.sh down qdrant

docker-down-chroma: ## Stop ChromaDB
	./docker/docker.sh down chroma

docker-down-neo4j: ## Stop Neo4j
	./docker/docker.sh down neo4j

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
	@echo "   - Milvus:   http://localhost:19530 (UI: http://localhost:9091)"
	@echo "   - Qdrant:   http://localhost:6333 (Dashboard: http://localhost:6333/dashboard)"
	@echo "   - ChromaDB: http://localhost:8000 (Docs: http://localhost:8000/docs)"
	@echo "   - Neo4j:    http://localhost:7474 (Browser, Bolt: bolt://localhost:7687)"
	@echo "   - Ollama:   http://localhost:11434"
	@echo ""
	@echo "Next steps:"
	@echo "   1. Run tests: make test"
	@echo "   2. Try examples: python examples/quickstart.py"
	@echo "   3. Check docs: make docs-serve"

dev-teardown: docker-down ## Stop all development services
	@echo "✅ Development services stopped"

# Examples
run-metadata-example: ## Run metadata extraction example (requires Ollama)
	@echo "🚀 Running metadata extraction example..."
	@echo "   Prerequisites: Ollama must be running with llama3.2 and nomic-embed-text models"
	@echo ""
	uv run python examples/metadata_extraction_example.py

run-graph-example: ## Run graph RAG example (requires Neo4j)
	@echo "🚀 Running Graph RAG example..."
	@echo "   Prerequisites: Neo4j must be running (make docker-up-neo4j)"
	@echo ""
	uv run python examples/graph_rag_basic.py

# Integration tests with Docker
test-integration: docker-up ## Run integration tests
	@echo "Waiting for services to be ready..."
	@sleep 10
	pytest tests/integration -v -m integration || true
	$(MAKE) docker-down
