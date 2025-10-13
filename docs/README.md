
# Documentation
This directory contains the **Sphinx documentation** for the `deckard` project.  
It includes API references, usage guides, and configuration examples for modules such as:

- [`deckard.data`](../deckard/data.py)
- [`deckard.model`](../deckard/model.py)
- [`deckard.attack`](../deckard/attack.py)

## Directory Structure

docs/
├── Makefile  
├── make.bat  
├── README.md ← this file   
└── source/  
├── conf.py ← Sphinx configuration  
├── index.rst ← main documentation entry point  
├── data.rst  
├── model.rst  
├── attack.rst  
└── _static/ ← optional static assets (CSS, images)  


## Prerequisites

```bash
pip install sphinx sphinx-rtd-theme phinx-autodoc-typehints sphinxcontrib-napoleon sphinx-autobuild
```
## Building the docs
Using make:
```bash
make html
```

Using python:
```
sphinx-build -b html source build/html
```

## Live Preview
To get a live preview of the docs, run:
```bash
sphinx-autobuild source build/html
```
which will open a webserver at [sphinx-build -b html source build/html
](sphinx-build -b html source build/html
) on your local machine.
