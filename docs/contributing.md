# Contributing to rag-toolkit

Thank you for your interest in contributing! This guide will help you get started.

## Code of Conduct

Please be respectful and considerate in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Set up development environment
4. Create a feature branch
5. Make your changes
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rag-toolkit.git
cd rag-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```

## Code Standards

### Type Hints

All functions must have type hints:

```python
# ✅ Good
def embed(self, text: str) -> list[float]:
    return [0.0] * 768

# ❌ Bad
def embed(self, text):
    return [0.0] * 768
```

### Docstrings

Use Google-style docstrings:

```python
def search(
    self,
    collection_name: str,
    query_vector: list[float],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Search for similar vectors in a collection.
    
    Args:
        collection_name: Name of the collection to search
        query_vector: Query embedding vector
        top_k: Number of results to return
        
    Returns:
        List of SearchResult objects sorted by relevance
        
    Raises:
        ValueError: If collection doesn't exist
        
    Example:
        >>> results = store.search("docs", query_vector, top_k=5)
        >>> print(len(results))
        5
    """
    ...
```

### Code Quality Tools

```bash
# Format code
make format

# Check linting
make lint

# Type checking
make typecheck

# Run all checks
make check
```

## Testing

### Writing Tests

```python
import pytest
from rag_toolkit import RagPipeline

def test_basic_query():
    """Test basic query functionality."""
    pipeline = RagPipeline(
        embedding_client=MockEmbedding(),
        llm_client=MockLLM(),
        vector_store=MockVectorStore(),
    )
    
    response = pipeline.query("test")
    assert response is not None
    assert isinstance(response.answer, str)

@pytest.mark.integration
def test_milvus_integration():
    """Integration test with real Milvus."""
    # Test with real services
    ...
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_pipeline.py::test_basic_query

# Run with coverage
make test-coverage

# Run integration tests (requires services)
pytest -m integration
```

## Pull Request Process

1. **Create an issue first** (for major changes)
2. **Create a branch**: `git checkout -b feature/my-feature`
3. **Make changes** following code standards
4. **Write tests** for new functionality
5. **Update documentation** if needed
6. **Run checks**: `make check`
7. **Commit**: Use conventional commits
8. **Push**: `git push origin feature/my-feature`
9. **Open PR** with clear description

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tooling changes

Example:
```
feat(vectorstore): add Pinecone implementation

Implements VectorStoreClient protocol for Pinecone.

Closes #123
```

## Adding New Features

### New Vector Store

1. Create file: `src/rag_toolkit/infra/vectorstores/your_store/client.py`
2. Implement `VectorStoreClient` protocol
3. Add tests: `tests/infra/vectorstores/test_your_store.py`
4. Update docs: `docs/user_guide/vector_stores.md`
5. Add example: `examples/custom_your_store.py`

```python
from rag_toolkit.core import VectorStoreClient
from rag_toolkit.core.types import SearchResult

class YourVectorStore:
    """Implementation for YourDB."""
    
    def create_collection(self, name: str, dimension: int, **kwargs) -> None:
        # Implementation
        ...
    
    # Implement all protocol methods
    ...
```

### New LLM Provider

1. Create file: `src/rag_toolkit/infra/llm/your_provider.py`
2. Implement `LLMClient` protocol
3. Add tests
4. Update documentation

### New Embedding Provider

1. Create file: `src/rag_toolkit/infra/embedding/your_provider.py`
2. Implement `EmbeddingClient` protocol
3. Add tests
4. Update documentation

## Documentation

### Building Docs

```bash
# Build documentation
make docs

# Serve locally
make docs-serve
# Open http://localhost:8000
```

### Adding Pages

1. Create `.md` file in `docs/`
2. Add to appropriate `.rst` toctree
3. Follow existing style
4. Build and verify

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.x.0`
4. Push tag: `git push origin v0.x.0`
5. GitHub Actions will publish to PyPI

## Getting Help

- 💬 [GitHub Discussions](https://github.com/gmottola00/rag-toolkit/discussions)
- 🐛 [Issue Tracker](https://github.com/gmottola00/rag-toolkit/issues)
- 📧 Email: gianmarcomottola00@gmail.com

## Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Attributed in relevant documentation

Thank you for contributing! 🎉
