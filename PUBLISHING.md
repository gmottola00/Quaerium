# Quaerium v0.1.2 - Publishing Guide

This document tracks the publication process for Quaerium v0.1.2, the first public release under the new package name.

## 📦 Release Summary

- **Version**: 0.1.2
- **Package Name**: quaerium (renamed from rag-toolkit)
- **Release Date**: 2026-02-10
- **Release Type**: First public release with major features

### Key Features in This Release

1. **Graph RAG with Neo4j**
   - GraphStoreClient protocol
   - Production-ready Neo4j 5.x client
   - Complete documentation and examples
   - 27 tests (unit + integration)

2. **Vector Store Integrations**
   - Qdrant and ChromaDB implementations
   - Benchmark framework

3. **Migration Tools**
   - VectorStoreMigrator for cross-store transfers
   - 60 tests with 100% pass rate

4. **Metadata Extraction & Enrichment**
   - LLMMetadataExtractor
   - MetadataEnricher
   - Batch embedding support

## ✅ Completed Steps

- [x] **Version updated** to 0.1.2 in `pyproject.toml`
- [x] **Changelog updated** in `docs/development/changelog.md`
- [x] **Package renamed** from rag-toolkit to quaerium (193 files changed)
- [x] **Changes committed** with comprehensive commit message
  - Commit: `2f1ae5000b655dfc12a2e114f3a42edb87a3d53f`
  - Message: "feat: release v0.1.2 - quaerium with graph RAG support"
- [x] **Package built successfully**
  - Wheel: `quaerium-0.1.2-py3-none-any.whl` (104KB)
  - Source: `quaerium-0.1.2.tar.gz` (75KB)
- [x] **Package metadata validated** with twine (PASSED)

## 🔑 PyPI Publication Steps

### Prerequisites

You need PyPI API tokens for authentication. Get them from:
- **TestPyPI**: https://test.pypi.org/manage/account/token/
- **PyPI**: https://pypi.org/manage/account/token/

### Option 1: Using ~/.pypirc (Recommended)

Create or update `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-PRODUCTION-API-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-API-TOKEN-HERE
```

**Security**: Set proper permissions:
```bash
chmod 600 ~/.pypirc
```

### Option 2: Using Environment Variables

```bash
# For TestPyPI
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR-TEST-TOKEN

# For PyPI
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR-PRODUCTION-TOKEN
```

### Publishing Workflow

#### 1. Publish to TestPyPI (Recommended First Step)

```bash
# Upload to TestPyPI
uv run twine upload --repository testpypi dist/*

# Expected output:
# Uploading distributions to https://test.pypi.org/legacy/
# Uploading quaerium-0.1.2-py3-none-any.whl
# Uploading quaerium-0.1.2.tar.gz
```

#### 2. Test Installation from TestPyPI

```bash
# Create a fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quaerium==0.1.2

# Test import
python -c "from quaerium import RagPipeline; from quaerium.infra.graphstores import create_neo4j_service; print('✅ Import successful!')"

# Test version
python -c "import quaerium; print(f'Version: {quaerium.__version__}')"

# Cleanup
deactivate
rm -rf test_env
```

#### 3. Publish to PyPI (Production)

```bash
# Final upload to PyPI
uv run twine upload dist/*

# Expected output:
# Uploading distributions to https://upload.pypi.org/legacy/
# Uploading quaerium-0.1.2-py3-none-any.whl
# Uploading quaerium-0.1.2.tar.gz
# View at:
# https://pypi.org/project/quaerium/0.1.2/
```

#### 4. Verify on PyPI

- **Package page**: https://pypi.org/project/quaerium/
- **Version page**: https://pypi.org/project/quaerium/0.1.2/

Test installation:
```bash
pip install quaerium==0.1.2
```

## 🚀 Post-Publication Steps

### 1. Push to GitHub

```bash
# Push the commit
git push origin main

# Create and push tag
git tag v0.1.2
git push origin v0.1.2
```

### 2. Create GitHub Release

Go to: https://github.com/gmottola00/quaerium/releases/new

**Tag**: `v0.1.2`

**Release Title**: `v0.1.2 - First Public Release with Graph RAG`

**Description** (use changelog content):

```markdown
# 🎉 Quaerium v0.1.2 - First Public Release

This is the first public release of **Quaerium** (formerly rag-toolkit), featuring comprehensive Graph RAG support and multiple enhancements.

## 🆕 What's New

### Graph RAG with Neo4j
- GraphStoreClient protocol for graph database operations
- Production-ready Neo4j 5.x client with async support
- Complete documentation and working examples
- 27 tests (unit + integration)

### Vector Store Integrations
- Qdrant integration with hybrid search
- ChromaDB support for local and client-server deployments
- Benchmark framework for performance comparison

### Migration Tools
- VectorStoreMigrator for cross-store data transfers
- Dry-run mode, filtered migration, retry logic
- 60 tests with 100% pass rate

### Metadata Extraction & Enrichment
- LLMMetadataExtractor for domain-specific extraction
- MetadataEnricher for chunk text enrichment
- Batch embedding support

## 📦 Installation

```bash
pip install quaerium

# With graph support
pip install quaerium[neo4j]

# With all features
pip install quaerium[all]
```

## 🔄 Breaking Changes

- Package renamed from `rag-toolkit` to `quaerium`
- Update imports: `from rag_toolkit` → `from quaerium`

## 📚 Documentation

Full documentation: https://gmottola00.github.io/quaerium/

## 🙏 Acknowledgments

Built with inspiration from LangChain, LlamaIndex, and Haystack.
```

### 3. Update Documentation Site

If using GitHub Pages:

```bash
# Build documentation
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### 4. Announce the Release

Consider announcing on:
- GitHub Discussions
- Twitter/X
- Reddit (r/Python, r/MachineLearning)
- LinkedIn
- Project blog

## 📋 Remaining Tasks

- [ ] **Task #16**: Publish to TestPyPI
- [ ] **Task #17**: Test installation from TestPyPI
- [ ] **Task #18**: Publish to PyPI official
- [ ] **Task #19**: Create GitHub Release (tag v0.1.2)
- [ ] **Task #20**: Deploy documentation to GitHub Pages
- [ ] **Task #21**: Announce release

## 🐛 Troubleshooting

### Authentication Issues

If you get authentication errors:

```bash
# Clear cached credentials
rm -rf ~/.pypi-credentials

# Use explicit token
uv run twine upload --username __token__ --password pypi-YOUR-TOKEN dist/*
```

### Package Already Exists

If the version already exists on PyPI:

```bash
# Bump version
# Edit pyproject.toml: version = "0.1.3"
# Edit docs/development/changelog.md
# Rebuild
rm -rf dist build src/*.egg-info
uv run python -m build
```

### Import Errors After Installation

Check installed version:
```bash
pip show quaerium
python -c "import quaerium; print(quaerium.__version__)"
```

Force reinstall:
```bash
pip uninstall quaerium -y
pip install --no-cache-dir quaerium==0.1.2
```

## 📝 Notes

- **Build artifacts** are in `dist/` directory
- **Built on**: 2026-02-24
- **Python versions**: 3.11, 3.12, 3.13
- **License**: MIT

## 🔗 Links

- **PyPI**: https://pypi.org/project/quaerium/
- **TestPyPI**: https://test.pypi.org/project/quaerium/
- **Documentation**: https://gmottola00.github.io/quaerium/
- **Repository**: https://github.com/gmottola00/quaerium
- **Issues**: https://github.com/gmottola00/quaerium/issues

---

**Last Updated**: 2026-02-24
