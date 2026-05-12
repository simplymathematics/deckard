Data
===========

The :mod:`deckard.data` module defines the :class:`~deckard.data.DataConfig` dataclass,
which provides a unified interface for loading, generating, preprocessing, and
splitting datasets for machine learning experiments.  
It supports both real and synthetic datasets, as well as YAML/Hydra-based configuration.

.. automodule:: deckard.data
   :members:
   :show-inheritance:

Data Sampling
-------------

The :mod:`deckard.data.sample` module provides pluggable sampling strategies via :class:`~deckard.data.sample.BaseSampler`
for robust train/test/validation splits.

.. automodule:: deckard.data.sample
   :members:
   :show-inheritance:

Data Preprocessing Pipelines
-----------------------------

The :class:`~deckard.data.DataPipelineConfig` wraps scikit-learn's :class:`~sklearn.pipeline.Pipeline`
to enable configurable feature preprocessing with timing instrumentation.

Extensions
----------

Pipeline Extension
~~~~~~~~~~~~~~~~~~

deckard exposes a configurable pipeline layer for data preprocessing via
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

:class:`~deckard.data.DataConfig` can load well-known datasets such as:

- **Adult Income** (via OpenML)
- **Diabetes** and **Digits** (from scikit-learn)
- **Synthetic datasets** via ``make_classification`` or ``make_regression``
- **CSV files** that contain a ``target`` column

It also supports **reproducible splits** via `train_test_split` with optional stratification,
timing instrumentation, and hashing for config tracking.

Data scoring mode
~~~~~~~~~~~~~~~~~

``DataConfig`` supports mode-aware dataset scoring via ``score_mode`` with
values:

- ``train``
- ``test``
- ``val``
- ``pre-sample``

``pre-sample`` runs data diagnostics against the full dataset before split
selection (``_X`` / ``_y``), while split modes run diagnostics on the selected
partition.

Examples
--------

.. seealso::

   Notebook-based examples for data loading, splitting, fairness data workflows,
   and PyTorch datasets are documented in:

   - :doc:`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - :doc:`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - :doc:`notebooks/pytorch.ipynb </notebooks/pytorch>`

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
* :doc:`model` — model configuration and training
* :doc:`experiment` — experiment orchestration
* :doc:`attack` — attack configuration
* :doc:`score` — scoring framework
* :doc:`pytorch` — PyTorch data integration
* :doc:`anjana` — anonymization-aware data
* :doc:`lifelines` — survival analysis data configuration
* :doc:`utils` — utility functions
