# :material-download: Installation

Get RAG Toolkit up and running in minutes with flexible installation options tailored to your needs.

---

## :material-check-circle: Requirements

!!! info "System Requirements"
    
    === "Minimum"
        - **Python**: 3.11 or higher
        - **Package Manager**: pip or conda
        - **Memory**: 2GB RAM minimum
        - **Disk**: 500MB free space
    
    === "Recommended"
        - **Python**: 3.12+
        - **Package Manager**: pip 24.0+
        - **Memory**: 8GB RAM (for LLM operations)
        - **Disk**: 5GB free space
        - **GPU**: CUDA-compatible (optional, for faster inference)
    
    === "Development"
        - All recommended requirements
        - **Git**: For version control
        - **Docker**: For running services
        - **Make**: For build automation

---

## :material-package: Basic Installation

!!! success "Quick Start"
    Install the core library with minimal dependencies:
    
    ```bash
    pip install rag-toolkit
    ```

**What's included:**

- :material-shield-check: Core protocols and types
- :material-code-braces: Pydantic models for validation
- :material-application-cog: Basic infrastructure components

---

## :material-puzzle: Optional Dependencies

!!! tip "Modular Installation"
    RAG Toolkit uses optional dependencies to keep the core library lightweight. Install only what you need!

### :material-robot: LLM Providers

<div class="grid cards" markdown>

- :material-server: **Ollama** — Local LLM server

    ---

    ```bash
    pip install rag-toolkit[ollama]
    ```

    Perfect for privacy-focused deployments and offline use.

- :material-openai: **OpenAI** — GPT models

    ---

    ```bash
    pip install rag-toolkit[openai]
    ```

    Access state-of-the-art models like GPT-4 and GPT-3.5.

</div>

### :material-file-document: Document Parsing

<div class="grid cards" markdown>

- :material-file-pdf-box: **PDF Support**

    ---

    ```bash
    pip install rag-toolkit[pdf]
    ```

    Parse PDF documents with advanced layout detection.

- :material-file-word: **DOCX Support**

    ---

    ```bash
    pip install rag-toolkit[docx]
    ```

    Extract content from Microsoft Word documents.

- :material-scanner: **OCR Support**

    ---

    ```bash
    pip install rag-toolkit[ocr]
    ```

    Extract text from scanned documents and images.

</div>

### :material-translate: Language Detection

```bash
pip install rag-toolkit[lang]
```

Automatic language detection for multilingual document processing.

---

### :material-package-variant: Combined Installations

!!! example "Common Combinations"
    
    === "All LLM Providers"
        ```bash
        pip install rag-toolkit[ollama,openai]
        ```
        
        Install support for both local and cloud LLM providers.
    
    === "All Document Parsers"
        ```bash
        pip install rag-toolkit[pdf,docx,ocr]
        ```
        
        Handle any document format: PDFs, Word docs, and scanned images.
    
    === "Everything"
        ```bash
        pip install rag-toolkit[all]
        ```
        
        Get the full RAG Toolkit experience with all optional features.

---

## :material-wrench: Development Installation

!!! warning "For Contributors"
    This installation method is for developers contributing to RAG Toolkit or running from source.

```bash title="Clone and Install"
# Clone the repository
git clone https://github.com/gmottola00/rag-toolkit.git
cd rag-toolkit

# Install in editable mode with dev dependencies
pip install -e ".[dev,docs]"
```

**Development tools included:**

<div class="grid" markdown>

- :material-test-tube: **Testing**: pytest, pytest-cov, pytest-asyncio
- :material-code-tags: **Type Checking**: mypy with strict mode
- :material-format-paint: **Formatting**: black, isort for consistent style
- :material-bug: **Linting**: ruff for fast Python linting
- :material-book-open: **Documentation**: MkDocs Material ecosystem
- :material-puzzle: **All Features**: All optional dependencies for comprehensive testing

</div>

---

## :material-check-all: Verification

!!! success "Test Your Installation"
    Run this script to verify everything is working correctly:

```python title="verify_installation.py" linenums="1" hl_lines="3-5"
# Test basic imports
from rag_toolkit import RagPipeline
from rag_toolkit.core import VectorStoreClient, EmbeddingClient, LLMClient

print("✅ Core installation verified!")

# Test optional imports (if installed)
try:
    from rag_toolkit import get_ollama_embedding
    print("✅ Ollama support available")
except ImportError:
    print("ℹ️  Ollama not installed (optional)")

try:
    from rag_toolkit import get_openai_embedding
    print("✅ OpenAI support available")
except ImportError:
    print("ℹ️  OpenAI not installed (optional)")
```

!!! tip "Expected Output"
    You should see at minimum:
    ```
    ✅ Core installation verified!
    ```
    
    Additional messages indicate which optional features are available.

---

## :material-cloud: External Services

### :material-database: Milvus (Vector Store)

!!! abstract "Default Vector Database"
    RAG Toolkit uses Milvus as the default vector store for high-performance similarity search.

=== "Docker Compose (Recommended)"
    ```bash
    docker-compose up -d milvus
    ```
    
    Starts Milvus with optimal production settings.

=== "Makefile"
    ```bash
    make docker-milvus
    ```
    
    Convenient shortcut for Docker Compose.

=== "Milvus Lite (Embedded)"
    ```bash
    pip install milvus
    ```
    
    Lightweight option for development and testing.

### :material-robot-outline: Ollama (Optional)

!!! info "Local LLM Server"
    Run large language models locally for privacy and cost savings.

=== "macOS/Linux"
    ```bash
    curl -fsSL https://ollama.ai/install.sh | sh
    ```

=== "Docker"
    ```bash
    docker run -d -p 11434:11434 ollama/ollama
    ```

=== "Pull Models"
    ```bash
    # Language model
    ollama pull llama2
    
    # Embedding model
    ollama pull nomic-embed-text
    ```

---

## :material-help-circle: Troubleshooting

!!! question "Common Issues"

### :material-alert: Import Errors

!!! failure "Problem"
    ```python
    ModuleNotFoundError: No module named 'ollama'
    ```

!!! success "Solution"
    Install the required optional dependency:
    ```bash
    pip install rag-toolkit[ollama]
    ```

### :material-type-check: Type Checking Issues

!!! failure "Problem"
    Mypy reports missing type stubs for third-party libraries.

!!! success "Solution"
    ```bash
    pip install types-requests types-urllib3
    ```

### :material-docker: Docker Issues

!!! failure "Problem"
    Milvus container won't start or connection refused.

!!! success "Solution"
    ```bash
    # Check if ports are available
    lsof -i :19530
    
    # Clean up and restart
    docker-compose down -v
    docker-compose up -d milvus
    
    # Check container logs
    docker logs milvus-standalone
    ```

---

## :material-compass: Next Steps

<div class="grid cards" markdown>

- :material-rocket-launch: **Quick Start**

    ---

    Build your first RAG application in 5 minutes

    [:material-arrow-right: Get Started](quickstart.md)

- :material-school: **Core Concepts**

    ---

    Learn the fundamental principles of RAG Toolkit

    [:material-arrow-right: Learn More](../guides/core_concepts.md)

- :material-code-braces: **Examples**

    ---

    Explore real-world RAG applications

    [:material-arrow-right: View Examples](../examples/index.md)

- :material-api: **API Reference**

    ---

    Complete API documentation

    [:material-arrow-right: Read Docs](../api/index.md)

</div>

---

## :material-laptop: Platform-Specific Notes

=== ":material-apple: macOS"
    
    !!! warning "Apple Silicon (M1/M2/M3)"
        Some dependencies may need Rosetta 2:
        ```bash
        softwareupdate --install-rosetta
        ```
    
    !!! tip "Homebrew Users"
        Consider installing Python via Homebrew for better compatibility:
        ```bash
        brew install python@3.12
        ```

=== ":material-linux: Linux"
    
    !!! info "Development Headers Required"
        Ensure you have Python development headers:
        
        ```bash title="Ubuntu/Debian"
        sudo apt-get update
        sudo apt-get install python3.11-dev build-essential
        ```
        
        ```bash title="CentOS/RHEL"
        sudo yum install python311-devel gcc gcc-c++
        ```
        
        ```bash title="Arch Linux"
        sudo pacman -S python python-pip base-devel
        ```

=== ":material-microsoft-windows: Windows"
    
    !!! warning "WSL2 Recommended"
        For best compatibility, use Windows Subsystem for Linux:
        ```bash
        wsl --install
        ```
    
    !!! info "Native Windows"
        If using native Windows:
        
        1. Install [Python from python.org](https://www.python.org/downloads/)
        2. Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/downloads/)
        3. Use PowerShell or Windows Terminal

---

## :material-update: Update Instructions

!!! tip "Stay Up to Date"
    Keep your RAG Toolkit installation current for the latest features and security patches.

=== "Core Library"
    ```bash
    pip install --upgrade rag-toolkit
    ```

=== "With All Features"
    ```bash
    pip install --upgrade rag-toolkit[all]
    ```

=== "Check Version"
    ```bash
    python -c "import rag_toolkit; print(rag_toolkit.__version__)"
    ```

!!! info "Release Notes"
    Check the [Changelog](../development/changelog.md) for what's new in each version.
