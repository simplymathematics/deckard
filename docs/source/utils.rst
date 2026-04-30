"""
utils.py
========

This module provides utility functions and classes for configuration management, data serialization, and command-line argument parsing.

Main Components
---------------

- **initialize_config**: Initializes and composes a Hydra configuration, supporting parameter overrides and dynamic target assignment.
- **ConfigBase**: A dataclass base class providing methods for saving/loading data, scores, and objects in various formats (CSV, JSON, Excel, Parquet, Pickle, HTML), as well as hashing and serialization utilities.
- **create_parser_from_function**: Generates an `argparse.ArgumentParser` from a function's signature, automatically mapping parameters to command-line arguments.

Features
--------

- Flexible configuration loading and instantiation using Hydra.
- Abstract base class for configuration objects with serialization and deserialization support.
- Convenient methods for saving/loading pandas DataFrames and score dictionaries in multiple formats.
- Automatic command-line parser generation from Python function signatures.

Dependencies
------------

- logging
- argparse
- inspect
- pathlib
- hashlib
- typing
- dataclasses
- pandas
- pickle
- hydra

Usage Example
-------------

.. code-block:: python

    # Initialize a configuration object
    config = initialize_config("config.yaml", params=["foo=1"], target="my.module.Class")

    # Save scores to a CSV file
    config.save_scores({"accuracy": 0.95}, "results/scores.csv")

    # Load data from a Parquet file
    df = config.load_data("data/dataset.parquet")

    # Create an argument parser from a function
    def train_model(epochs: int, lr: float = 0.01):
        pass

    parser = create_parser_from_function(train_model)
    args = parser.parse_args()

"""

# ConfigBase

"""
ConfigBase
==========

A base dataclass providing utility methods for configuration management, data persistence, and serialization.

Attributes
----------
_target_ : str
    The target string used for Hydra instantiation. Defaults to "deckard.utils.ConfigBase".

Methods
-------
__init__(*args, **kwds)
    Initializes the dataclass and sets attributes from positional and keyword arguments.

__post_init__()
    Optional post-initialization hook for subclasses.

__call__(*args, **kwds)
    Abstract method. Raises NotImplementedError.

__hash__()
    Computes an MD5 hash based on all non-private attributes.

save_scores(scores, filepath=None)
    Saves a dictionary or pandas Series of scores to a file (CSV, JSON, or XLSX).

save_data(data, filepath=None, **kwargs)
    Saves a pandas DataFrame to a file (CSV, Parquet, Pickle, HTML, JSON, or XLSX).

load_scores(filepath)
    Loads scores from a file (CSV, JSON, or XLSX) into a dictionary.

load_data(filepath, **kwargs)
    Loads data from a file (CSV, JSON, XLSX, Parquet, Pickle, NPZ, or HTML) into a pandas DataFrame.

save_object(obj, filepath)
    Serializes and saves an object to a file using pickle.

load_object(filepath)
    Loads a serialized object from a file using pickle.

save(filepath)
    Saves the current instance to a file using pickle.

load(filepath)
    Loads an instance from a file using pickle and updates the current instance.

Notes
-----
- Designed for extensibility and integration with Hydra configuration management.
- Provides robust file I/O for common data science formats.
- Ensures type safety and error handling for file operations.

Usage Example
-------------
.. code-block:: python

    from deckard.utils import ConfigBase
    import pandas as pd

    class MyConfig(ConfigBase):
        param1: int = 10
        param2: str = "default"

        def __post_init__(self):
            print(f"Initialized with param1={self.param1}, param2={self.param2}")

        def __call__(self, *args, **kwds):
            return self.param1 * 2

    config = MyConfig(param1=5)
    print(config())  # Outputs: 10

    # Save and load scores
    scores = {"accuracy": 0.95, "loss": 0.05}
    config.save_scores(scores, "scores.csv")
    loaded_scores = config.load_scores("scores.csv")
    print(loaded_scores)

    # Save and load data
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    config.save_data(df, "data.parquet")
    loaded_df = config.load_data("data.parquet")
    print(loaded_df)

    # Save and load model
    model = SomeModel()
    config.save_object(model, "model.pkl")
    loaded_model = config.load_object("model.pkl")
