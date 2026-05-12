# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build
NOTEBOOKSDIR  ?= notebooks
DVC           ?= dvc
DVC_REPRO_ARGS ?=
DVC_STAGE     ?=
SPHINX_NB_OFF ?= -D nb_execution_mode=off

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Extra targets:"
	@echo "  notebooks             Run DVC notebook pipeline in $(NOTEBOOKSDIR)"
	@echo "  html-with-notebooks   Run DVC notebooks, then build html with MyST execution disabled"
	@echo ""
	@echo "Optional variables:"
	@echo "  DVC_STAGE=<stage>       Run one DVC stage (for example: notebook_fairlearn)"
	@echo "  DVC_REPRO_ARGS='...'    Extra args forwarded to 'dvc repro'"

.PHONY: help Makefile notebooks html-with-notebooks clean

notebooks:
	@command -v "$(DVC)" >/dev/null 2>&1 || { \
		echo "Error: dvc executable not found. Install DVC or run 'make html' for Sphinx-only docs."; \
		exit 1; \
	}
	@cd "$(NOTEBOOKSDIR)" && $(DVC) repro $(DVC_REPRO_ARGS) $(DVC_STAGE)

html-with-notebooks: notebooks
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(SPHINX_NB_OFF) $(O)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@rm -rf "$(SOURCEDIR)/notebooks/build"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
