# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?= --keep-going --color
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PYTHON        ?= python

# Colors for terminal output
BOLD          := \033[1m
GREEN         := \033[0;32m
BLUE          := \033[0;34m
YELLOW        := \033[0;33m
RESET         := \033[0m

.PHONY: help clean html linkcheck serve watch all

help:
	@echo "$(BOLD)rag-toolkit Documentation$(RESET)"
	@echo ""
	@echo "$(GREEN)Available targets:$(RESET)"
	@echo "  $(BLUE)make html$(RESET)       - Build HTML documentation"
	@echo "  $(BLUE)make clean$(RESET)      - Remove build artifacts"
	@echo "  $(BLUE)make linkcheck$(RESET)  - Check all external links"
	@echo "  $(BLUE)make serve$(RESET)      - Build and serve docs locally"
	@echo "  $(BLUE)make watch$(RESET)      - Build and auto-rebuild on changes"
	@echo "  $(BLUE)make all$(RESET)        - Clean, build, and check links"
	@echo ""

clean:
	@echo "$(YELLOW)🧹 Cleaning build artifacts...$(RESET)"
	rm -rf $(BUILDDIR)/*
	rm -rf autoapi/
	@echo "$(GREEN)✅ Clean complete$(RESET)"

html:
	@echo "$(YELLOW)📚 Building HTML documentation...$(RESET)"
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@touch $(BUILDDIR)/html/.nojekyll
	@echo ""
	@echo "$(GREEN)✅ Build finished. HTML pages are in $(BUILDDIR)/html$(RESET)"

linkcheck:
	@echo "$(YELLOW)🔗 Checking links...$(RESET)"
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo "$(GREEN)✅ Link check complete$(RESET)"

serve: html
	@echo "$(GREEN)📚 Serving documentation at http://localhost:8000$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	@cd $(BUILDDIR)/html && $(PYTHON) -m http.server 8000

watch:
	@echo "$(GREEN)👀 Watching for changes... (Ctrl+C to stop)$(RESET)"
	@while true; do \
		$(MAKE) html 2>&1 | grep -v "reading sources"; \
		sleep 2; \
	done

all: clean html linkcheck
	@echo "$(GREEN)✅ Full build complete!$(RESET)"

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
