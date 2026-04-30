# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "deckard"
copyright = "2025, simplymathematics"
author = "simplymathematics"
release = ".91"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/Numpy style docstrings
    "sphinx.ext.viewcode",  # Adds source links
    "sphinx_autodoc_typehints",  # Render type hints
    "myst_parser",  # (optional) Markdown support
]

templates_path = []
exclude_patterns = []

# HTML theme
html_theme = "sphinx_rtd_theme"
html_static_path = []

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# napoleon_use_param = True
# napoleon_use_ivar = True
