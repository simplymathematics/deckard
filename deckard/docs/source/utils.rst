
"""
utils.py
=========

This module provides utility functions and classes for configuration management, data serialization, and command-line argument parsing in the Deckard project.

Main Components
---------------

- **ConfigBase**: An abstract base class for configuration objects, supporting serialization, deserialization, and data I/O operations with pandas and pickle. It includes methods for saving/loading scores and data in various formats (CSV, JSON, Excel, Parquet, Pickle, HTML), as well as saving/loading the configuration object itself.

- **initialize_config**: A function to initialize and compose Hydra configurations, supporting parameter overrides and dynamic target assignment.

- **create_parser_from_function**: A utility to generate an `argparse.ArgumentParser` from a function's signature, automatically mapping parameters to command-line arguments.

Features
--------

- Flexible configuration loading and instantiation using Hydra.
- Abstract configuration base class with hashing, serialization, and file I/O support.
- Automated command-line parser generation from function signatures.
- Logging for all major operations to aid debugging and traceability.

Dependencies
------------

- `logging`
- `argparse`
- `inspect`
- `pathlib`
- `hashlib`
- `dataclasses`
- `pandas`
- `pickle`
- `hydra`

Usage
-----

Import the module and use the provided utilities for configuration management, data handling, and command-line interface generation in your Deckard workflows.

"""
