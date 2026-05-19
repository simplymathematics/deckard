# Configuration file for the Sphinx documentation builder.
# Covers rst API docs and Jupyter notebooks (in notebooks/).

import os
import re
import sys
from pathlib import Path

# deckard package lives one level above docs/
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "deckard"
copyright = "2026, simplymathematics"
author = "simplymathematics"
release = "0.98.1"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

# Allow optional/heavy dependencies to be absent during docs builds
autodoc_mock_imports = [
    "fairlearn",
    "torch",
    "torchvision",
    "torchaudio",
]

templates_path = []

exclude_patterns = [
    "build",
    "**.ipynb_checkpoints",
    "notebooks/build",
    "notebooks/dvc.lock",
    "notebooks/dvc.yaml",
    "notebooks/error.log",
    "notebooks/deckard.log",
]

root_doc = "index"

# Notebook markdown cells intentionally start at H2 in many guides.
# Suppress style-only heading warnings during strict CI docs builds.
suppress_warnings = ["myst.header"]

# ---------------------------------------------------------------------------
# myst-nb settings
# ---------------------------------------------------------------------------
nb_execution_mode = "cache"
nb_execution_timeout = 1800
nb_execution_raise_on_error = False
nb_execution_excludepatterns = [
    "deckard.ipynb",
]

nb_kernel_name = "python3"
nb_render_plugin = "default"
nb_append_css = False

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = []

html_theme_options = {
    # Top header bar
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    # Left sidebar: expand two levels
    "show_nav_level": 1,
    "navigation_depth": 2,
    # Right sidebar: current page headings only
    "secondary_sidebar_items": ["page-toc"],
}

html_sidebars = {
    "**": [
        "sidebar-nav-bs.html",
    ],
}

# ---------------------------------------------------------------------------
# Napoleon settings
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Keep type hints readable in generated API docs without third-party
# typehint post-processing that can leak raw rst tokens.
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
napoleon_use_rtype = False


# ---------------------------------------------------------------------------
# Post-build sanitization
# Removes absolute local filesystem paths from generated docs.
# ---------------------------------------------------------------------------
def _sanitize_text_file(file_path: Path, pattern: str, replacement: str) -> bool:
    try:
        text = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False

    new_text, replacements = re.subn(pattern, replacement, text)

    if replacements == 0:
        return False

    file_path.write_text(new_text, encoding="utf-8")
    return True


def _sanitize_build_outputs(app, exception) -> None:
    if exception is not None:
        return

    repo_root_name = Path(__file__).resolve().parent.parent.name

    # Remove any absolute prefix before "<repo_root>/"
    pattern = rf"/[^\s\"'<>]*/(?={re.escape(repo_root_name)}/)"
    replacement = ""

    candidate_roots = [
        Path(app.outdir),
        Path(app.outdir).parent / "jupyter_execute",
        Path(app.outdir).parent / ".jupyter_cache",
        Path(app.outdir) / "reports",
    ]

    text_suffixes = {
        ".html",
        ".js",
        ".txt",
        ".log",
        ".json",
        ".ipynb",
        ".md",
        ".rst",
    }

    for root in candidate_roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in text_suffixes:
                _sanitize_text_file(path, pattern, replacement)


def setup(app):
    app.connect("build-finished", _sanitize_build_outputs)
