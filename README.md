# Deckard

Deckard is a Python application designed to [briefly describe purpose]. This README provides installation instructions for Windows, macOS, and Linux using both `pyenv` and Python's built-in `venv`.

## Prerequisites

- Python 3.10 or higher
- Git

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:simplymathematics/deckard.git
cd deckard
```

### 2. Setup a virtual environment

You can use either `pyenv` or `venv`. Below, there are instructions for both:

To set up a virtual environment using `pyenv`, follow these steps:

```bash
pyenv install 3.10
pyenv virtualenv 3.10 env
pyenv activate env
```

- `pyenv install 3.10` downloads and installs the newest Python version compatible with 3.10.
- `pyenv virtualenv 3.10 env` creates a new virtual environment named `env` using Python 3.10.
- `pyenv activate env` activates the `env` environment, so all Python commands use this isolated setup.

To set up a virtual environment using Python's built-in `venv`, run:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

- `python3 -m venv env` creates a new virtual environment named `env`.
- `source env/bin/activate` (or `env\Scripts\activate` on Windows) activates the environment so all Python commands use this isolated setup.

This command switches your shell to use the `env` environment, ensuring all Python packages are installed locally within it.



### 3. Install Dependencies

```bash
python -m pip install -e .
```

This command installs the project's dependencies in "editable" mode. Editable mode (`-e .`) allows you to modify the source code and have changes reflected immediately without needing to reinstall the package. The `-m` flag tells Python to run the `pip` module as a script, ensuring you use the correct version of `pip` for your environment.

## Usage

See [docs/README.md](docs/README.md) for detailed documentation.
