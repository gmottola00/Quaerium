# rag-toolkit Documentation

Professional Sphinx documentation for rag-toolkit.

## Quick Start

### Install Dependencies

```bash
pip install -e ".[docs]"
```

### Build Documentation

```bash
cd docs
make html
```

### View Locally

```bash
make serve
# Open http://localhost:8000
```

## Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst               # Homepage
├── installation.md         # Installation guide
├── quickstart.md           # Quick start tutorial
├── architecture.md         # Architecture overview
├── user_guide/             # User guide
│   ├── index.md
│   ├── core_concepts.md
│   ├── protocols.md
│   ├── vector_stores.md
│   ├── embeddings.md
│   ├── llms.md
│   ├── rag_pipeline.md
│   ├── chunking.md
│   └── reranking.md
├── examples/               # Examples
│   ├── index.md
│   ├── basic_rag.md
│   ├── custom_vectorstore.md
│   ├── hybrid_search.md
│   ├── advanced_pipeline.md
│   └── production_setup.md
├── contributing.md         # Contribution guide
├── changelog.md            # Version history
├── roadmap.md              # Future plans
├── _static/                # Static assets
│   └── custom.css          # Custom styling
└── _build/                 # Generated documentation
    └── html/               # HTML output
```

## Features

- ✅ **Furo Theme**: Modern, responsive design
- ✅ **sphinx-autoapi**: Automatic API documentation
- ✅ **MyST Parser**: Markdown support
- ✅ **Copy Buttons**: Easy code copying
- ✅ **GitHub Integration**: Edit links, source repository
- ✅ **Intersphinx**: Cross-project references
- ✅ **Custom CSS**: Branded styling

## Building

### Full Build

```bash
make html
```

### Clean Build

```bash
make clean
make html
```

### Check Links

```bash
make linkcheck
```

### Watch Mode

```bash
make watch  # Rebuilds on file changes
```

## Deployment

Documentation is automatically deployed to GitHub Pages on push to `main`:

1. GitHub Actions builds docs
2. Deploys to `gh-pages` branch
3. Available at: https://gmottola00.github.io/rag-toolkit/

### Manual Deployment

```bash
cd docs
make html
cd _build/html
touch .nojekyll  # Disable Jekyll
# Upload to hosting
```

## Writing Documentation

### Adding a New Page

1. Create `.md` file in appropriate directory
2. Add to toctree in relevant `index.rst` or `index.md`
3. Follow existing style and format
4. Build and verify

### Code Examples

Use syntax highlighting:

\`\`\`python
from rag_toolkit import RagPipeline

pipeline = RagPipeline(...)
\`\`\`

### Admonitions

\`\`\`markdown
:::{note}
This is a note
:::

:::{warning}
This is a warning
:::

:::{tip}
This is a tip
:::
\`\`\`

### Cross-References

- Link to API: \`{class}\`rag_toolkit.core.VectorStoreClient\`\`
- Link to page: \`[Installation](installation.md)\`
- Link to section: \`[Core Concepts](user_guide/core_concepts.md#embeddings)\`

## Troubleshooting

### Missing Module Errors

```bash
pip install -e ".[docs]"
```

### Build Errors

```bash
make clean
rm -rf autoapi/
make html
```

### Broken Links

```bash
make linkcheck
```

## Contributing

See [CONTRIBUTING.md](contributing.md) for:
- Writing documentation
- Adding examples
- Improving existing docs
- Translation

## Style Guide

- Use **Markdown** for content pages
- Use **reStructuredText** only for index files with toctrees
- Keep paragraphs concise
- Use code examples liberally
- Add emoji sparingly (main headings only)
- Include working code snippets

## Credits

- Theme: [Furo](https://pradyunsg.me/furo/)
- Generator: [Sphinx](https://www.sphinx-doc.org/)
- Hosting: [GitHub Pages](https://pages.github.com/)
