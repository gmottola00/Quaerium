# GitHub Pages Configuration for rag-toolkit

This document describes the GitHub Pages setup for rag-toolkit documentation.

## 🌐 GitHub Pages URL

**Documentation URL**: https://gmottola00.github.io/rag-toolkit/

## ⚙️ Repository Settings

### Enable GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Under "Build and deployment":
   - **Source**: Deploy from a branch
   - **Branch**: `gh-pages`
   - **Folder**: `/ (root)`
3. Click **Save**

### Required Permissions

The workflow needs these permissions (already configured in `docs.yml`):

```yaml
permissions:
  contents: read
  pages: write
  id-token: write
```

## 🚀 Deployment Workflow

### Automatic Deployment

Documentation is automatically deployed when:

- **Push to `main` branch** → Build + Deploy
- **Push to `develop` branch** → Build only (no deploy)
- **Pull Request** → Build only (no deploy)

### Manual Deployment

Trigger manual deployment:

1. Go to **Actions** tab
2. Select "Build and Deploy Sphinx Documentation"
3. Click **Run workflow**
4. Select branch (`main`)
5. Click **Run workflow**

## 📋 Workflow Steps

### Build Job

1. **Checkout**: Clone repository with full history
2. **Setup Python**: Install Python 3.11
3. **Cache**: Cache pip packages for faster builds
4. **Install**: Install package with `[docs]` extras
5. **Verify**: Verify package imports work correctly
6. **Build**: Build Sphinx documentation with warnings as errors
7. **Check Links**: Check for broken external links
8. **Upload**: Upload HTML artifacts to GitHub Pages

### Deploy Job

1. **Deploy**: Deploy HTML to GitHub Pages
2. **Summary**: Display deployment URL

## 🔍 Build Configuration

### Sphinx Options

```makefile
SPHINXOPTS = --keep-going --color
```

- `--keep-going`: Continue on errors
- `--color`: Colored output

### Environment Variables

```yaml
SPHINX_BUILD: "true"  # Signals documentation build mode
```

## 📦 Dependencies

Documentation dependencies from `pyproject.toml`:

```toml
[project.optional-dependencies]
docs = [
    "sphinx>=7.0.0,<8.0.0",
    "furo>=2024.0.0",
    "sphinx-autoapi>=3.0.0,<4.0.0",
    "myst-parser>=2.0.0,<3.0.0",
    "sphinx-copybutton>=0.5.0",
    "sphinx-design>=0.5.0",
]
```

Install with:

```bash
pip install -e ".[docs]"
```

## 🔧 Troubleshooting

### Build Fails

**Problem**: Build fails in GitHub Actions

**Solutions**:

1. **Check workflow logs** in Actions tab
2. **Test locally**:
   ```bash
   cd docs
   make clean
   make html
   ```
3. **Verify dependencies**:
   ```bash
   pip install -e ".[docs]"
   python -c "import rag_toolkit; print('OK')"
   ```

### Pages Not Updating

**Problem**: GitHub Pages shows old content

**Solutions**:

1. **Check deployment status** in Actions tab
2. **Clear browser cache**: Ctrl+Shift+R (Chrome/Firefox)
3. **Check branch**: Ensure `gh-pages` branch exists
4. **Verify workflow**: Check workflow completed successfully

### 404 Error

**Problem**: Pages show 404 error

**Solutions**:

1. **Check .nojekyll file** exists in root:
   ```bash
   git checkout gh-pages
   ls -la .nojekyll
   ```
2. **Verify index.html** exists:
   ```bash
   ls index.html
   ```
3. **Check Pages settings** in repository

### Import Errors

**Problem**: Package imports fail during build

**Solutions**:

1. **Verify package structure**:
   ```bash
   python -c "import rag_toolkit; print(rag_toolkit.__version__)"
   ```
2. **Check PYTHONPATH**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```
3. **Install in editable mode**:
   ```bash
   pip install -e .
   ```

## 📊 Build Status

### Current Status

- ✅ Build Time: ~2-3 minutes
- ✅ Warnings: 82 (API documentation only)
- ✅ Link Check: Passing
- ✅ Size: ~15 MB

### Optimization Tips

1. **Use caching**: Workflow caches pip packages
2. **Parallel builds**: sphinx-autoapi builds in parallel
3. **Skip unchanged**: Only builds on docs/ changes

## 🔐 Security

### Secrets

No secrets required for documentation deployment.

### Permissions

Workflow uses GitHub's built-in `GITHUB_TOKEN` with minimal permissions:

- `contents: read` - Read repository
- `pages: write` - Write to GitHub Pages
- `id-token: write` - OIDC token for Pages

## 📈 Monitoring

### View Deployments

1. Go to **Actions** tab
2. Select workflow run
3. View logs and status
4. Check deployment URL

### Deployment History

1. Go to **Settings** → **Pages**
2. View deployment history
3. See active deployments

## 🔄 Updates

### Update Documentation

1. Edit documentation files in `docs/`
2. Commit and push:
   ```bash
   git add docs/
   git commit -m "docs: update documentation"
   git push origin main
   ```
3. GitHub Actions automatically rebuilds and deploys

### Update Workflow

1. Edit `.github/workflows/docs.yml`
2. Test changes:
   ```bash
   # Validate YAML
   yamllint .github/workflows/docs.yml
   ```
3. Commit and push:
   ```bash
   git add .github/workflows/docs.yml
   git commit -m "ci: update docs workflow"
   git push
   ```

## 📚 Resources

- [GitHub Pages Documentation](https://docs.github.com/pages)
- [GitHub Actions Documentation](https://docs.github.com/actions)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Deploy Pages Action](https://github.com/actions/deploy-pages)

## 🎯 Next Steps

After setup:

1. ✅ Enable GitHub Pages in repository settings
2. ✅ Push to main branch to trigger first deployment
3. ✅ Verify deployment at https://gmottola00.github.io/rag-toolkit/
4. ✅ Add documentation badge to README.md
5. ✅ Share documentation URL with users

## 📝 Notes

- **First deployment** takes 5-10 minutes
- **Subsequent deploys** take 2-3 minutes
- **Pages update** within 1-2 minutes after deployment
- **Custom domain** can be configured in repository settings
- **HTTPS** is automatically enabled by GitHub Pages

---

For questions or issues, open an issue on GitHub.
