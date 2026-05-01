Data
===========

The :mod:`deckard.data` module defines the :class:`~deckard.data.DataConfig` dataclass,
which provides a unified interface for loading, generating, preprocessing, and
splitting datasets for machine learning experiments.  
It supports both real and synthetic datasets, as well as YAML/Hydra-based configuration.

.. automodule:: deckard.data
   :members:
   :show-inheritance:

Extensions
----------

Pipeline Extension
~~~~~~~~~~~~~~~~~~

Deckard exposes a configurable pipeline layer for data preprocessing via
:class:`~deckard.data.DataPipelineConfig`.

Fairness Extension
~~~~~~~~~~~~~~~~~~

The fairness extension adds group-aware sampling and fairness metrics with
``fairlearn`` integration.

.. automodule:: deckard.data.fairness
   :members:
   :show-inheritance:

Torch Extension
~~~~~~~~~~~~~~~

The torch extension provides dataset loading and sampling for PyTorch and
torchvision-backed workflows.

.. automodule:: deckard.data.pytorch
   :members:
   :show-inheritance:

Survival Extension
------------------

Survival-specific experiment orchestration is split into a dedicated optional
module.

.. automodule:: deckard.data.survival
   :members:
   :show-inheritance:

Overview
--------

Overview
--------

:class:`~deckard.data.DataConfig` can load well-known datasets such as:

- **Adult Income** (via OpenML)
- **Diabetes** and **Digits** (from scikit-learn)
- **Synthetic datasets** via ``make_classification`` or ``make_regression``
- **CSV files** that contain a ``target`` column

It also supports **reproducible splits** via `train_test_split` with optional stratification,
timing instrumentation, and hashing for config tracking.

Usage
-----

Command-line example
~~~~~~~~~~~~~~~~~~~~

Run data setup directly from the terminal:

.. code-block:: bash

   # Load the Adult dataset with default settings
   python -m deckard optimize --config-name experiment data.dataset_name=adult data.test_size=0.2

   # Override configuration parameters inline
   python -m deckard optimize --config-name experiment data.dataset_name=make_classification data.test_size=0.25

Programmatic usage
~~~~~~~~~~~~~~~~~~

Use :class:`~deckard.data.DataConfig` from within your Python scripts or notebooks:

.. code-block:: python

   from deckard.data import initialize_data_config

   # Initialize using default, Hydra/YAML, or CLI configuration
   data = initialize_data_config(dataset_name="adult", test_size=0.2, random_state=42)

   # Load and split the dataset
   data(data_filepath="data.pkl", score_file="data_scores.json")
   X_train = data.X_train
   X_test = data.X_test
   y_train = data.y_train
   y_test = data.y_test

   print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

Custom configuration
~~~~~~~~~~~~~~~~~~~~

You can define a YAML file or override config parameters inline.

Example minimal YAML (`adult.yaml`):

.. code-block:: yaml

   _target_: data.DataConfig
   dataset_name: adult
   test_size: 0.2
   random_state: 42
   stratify: True

Example inline overrides:

.. code-block:: bash

   python -m deckard optimize --config-name experiment data.dataset_name=make_classification data.data_params.n_samples=2000 data.data_params.n_features=20

Internals
---------

Timing and logging
~~~~~~~~~~~~~~~~~~
The data loading and splitting process is timed, and the duration is stored in
the `_data_load_time` and `_data_sample_time` attributes of the :class:`~deckard.data.DataConfig` instance. This can be useful for comparing the run-time efficiency of different datasets of various methods. 
Logging is performed at key steps.


Troubleshooting
---------------
If you encounter issues with dataset loading, ensure that:
- You have an active internet connection for datasets fetched from OpenML.
- The specified CSV file path is correct and the file is accessible.
- Otherwise, use one of the built-in datasets or synthetic data generation options.

See also
~~~~~~~~
* :doc:`attack`
* :doc:`model`
* :doc:`experiment`
* :doc:`utils`
