#!/bin/bash
# Update benchmark report in Sphinx documentation

set -e

echo "🔄 Updating benchmark report in documentation..."

# Check if benchmark report exists
if [ ! -f "benchmark_report.html" ]; then
    echo "❌ benchmark_report.html not found"
    echo "   Run 'make benchmark-report' first"
    exit 1
fi

# Ensure _static directory exists
mkdir -p docs/_static

# Copy report to _static
cp benchmark_report.html docs/_static/
echo "✅ Copied benchmark_report.html to docs/_static/"

# Generate timestamp file
cat > docs/_static/benchmark_timestamp.txt << EOF
Last updated: $(date '+%Y-%m-%d %H:%M:%S')
EOF

echo "✅ Documentation updated!"
echo ""
echo "Next steps:"
echo "  1. cd docs && make html"
echo "  2. The report will be at: _build/html/_static/benchmark_report.html"
echo ""
echo "Or for GitHub Pages:"
echo "  1. git add docs/_static/benchmark_report.html"
echo "  2. git commit -m 'Update benchmark report'"
echo "  3. git push"
