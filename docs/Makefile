# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= $(shell if [ -x ../.venv/bin/sphinx-build ]; then echo ../.venv/bin/sphinx-build; else echo sphinx-build; fi)
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = build
NOTEBOOKSDIR  ?= notebooks
DVC           ?= dvc
DVC_REPRO_ARGS ?=
DVC_STAGE     ?=
SPHINX_NB_OFF ?= -D nb_execution_mode=off
AUTOBUILD_PORT ?= 8000
AUTOBUILD_OPTS ?= -j auto --port $(AUTOBUILD_PORT)
AUTOBUILD_WATCH ?= --watch ../deckard --watch ../examples --watch $(NOTEBOOKSDIR)/dvc.yaml

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Extra targets:"
	@echo "  html                  Run DVC notebooks, then build html with MyST execution disabled"
	@echo "  autobuild             Run sphinx-autobuild with DVC notebook repro as a pre-build step"
	@echo "  notebooks             Run DVC notebook pipeline in $(NOTEBOOKSDIR)"
	@echo "  html-with-notebooks   Alias for html"
	@echo ""
	@echo "Optional variables:"
	@echo "  DVC_STAGE=<stage>       Run one DVC stage (for example: notebook_fairlearn)"
	@echo "  DVC_REPRO_ARGS='...'    Extra args forwarded to 'dvc repro'"
	@echo "  AUTOBUILD_PORT=<port>   Port used by sphinx-autobuild (default: $(AUTOBUILD_PORT))"
	@echo "  AUTOBUILD_OPTS='...'    Extra args forwarded to sphinx-autobuild"

.PHONY: help Makefile notebooks html html-with-notebooks autobuild clean

notebooks:
	@command -v "$(DVC)" >/dev/null 2>&1 || { \
		echo "Error: dvc executable not found. Install DVC or run 'make html' for Sphinx-only docs."; \
		exit 1; \
	}
	@cd "$(NOTEBOOKSDIR)" && $(DVC) pull || echo "DVC pull: no remote configured or cache empty; proceeding with pipeline"
	@cd "$(NOTEBOOKSDIR)" && $(DVC) repro $(DVC_REPRO_ARGS) $(DVC_STAGE)
	@cd "$(NOTEBOOKSDIR)" && $(DVC) push || echo "DVC push: remote not configured; cache will be managed locally"

html: notebooks
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(SPHINX_NB_OFF) $(O)

html-with-notebooks: notebooks
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(SPHINX_NB_OFF) $(O)

autobuild:
	@command -v "$(SPHINXAUTOBUILD)" >/dev/null 2>&1 || { \
		echo "Error: sphinx-autobuild executable not found. Install it with 'pip install sphinx-autobuild'."; \
		exit 1; \
	}
	@$(SPHINXAUTOBUILD) $(AUTOBUILD_OPTS) \
		--pre-build '$(MAKE) notebooks DVC="$(DVC)" DVC_REPRO_ARGS="$(DVC_REPRO_ARGS)" DVC_STAGE="$(DVC_STAGE)"' \
		$(AUTOBUILD_WATCH) \
		$(SPHINX_NB_OFF) \
		"$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@rm -rf "$(SOURCEDIR)/notebooks/build"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
