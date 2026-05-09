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

Usage
-----

Command-line example
~~~~~~~~~~~~~~~~~~~~

Run data setup directly from the terminal:

.. code-block:: bash

   # Integration-style synthetic classification data
   python -m deckard optimize --config-name experiment \
      data.dataset_name=make_classification \
      data.data_params.n_samples=40 \
      data.data_params.n_features=10 \
      data.train_size=30 \
      data.test_size=10

   # Fairness integration-style Adult split
   python -m deckard optimize --config-name experiment \
      data.dataset_name=adult \
      data.train_size=160 \
      data.test_size=80

Programmatic usage
~~~~~~~~~~~~~~~~~~

.. seealso::

   Fully-executed programmatic examples — including classification, regression,
   fairness-aware data, and survival data — are available in the
   :doc:`notebooks/sklearn.ipynb </notebooks/sklearn>` notebook.

Fairness data usage
~~~~~~~~~~~~~~~~~~~

The :class:`~deckard.data.fairness.FairlearnDataConfig` adds group-aware
sampling and fairness metrics with ``fairlearn`` integration.
See the :doc:`notebooks/fairlearn.ipynb </notebooks/fairlearn>` notebook for an executed example.

Survival data usage
~~~~~~~~~~~~~~~~~~~

Survival integrations use :class:`~deckard.data.DataConfig` for native
lifelines datasets. See the :doc:`notebooks/sklearn.ipynb </notebooks/sklearn>` notebook for examples.

Custom configuration
~~~~~~~~~~~~~~~~~~~~

You can define a YAML file or override config parameters inline.

Example minimal YAML (`adult.yaml`):

.. code-block:: yaml

   _target_: deckard.data.DataConfig
   dataset_name: make_classification
   data_params:
     n_samples: 40
     n_features: 10
     n_informative: 4
     n_redundant: 0
     n_clusters_per_class: 1
     n_classes: 2
     random_state: 17
   train_size: 30
   test_size: 10
   random_state: 42
   stratify: True
   classifier: True

Example inline overrides:

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data.dataset_name=make_classification \
      data.data_params.n_samples=40 \
      data.data_params.n_features=10 \
      data.train_size=30 \
      data.test_size=10

Data Pipeline Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply scikit-learn preprocessing pipelines via :class:`~deckard.data.DataPipelineConfig`:

.. code-block:: python

   from deckard.data import DataConfig, DataPipelineConfig

   # Create a data config with preprocessing pipeline
   data = DataConfig(
      dataset_name="make_classification",
      data_params={
         "n_samples": 100,
         "n_features": 20,
         "n_informative": 10,
         "random_state": 42,
      },
      train_size=70,
      test_size=30,
      classifier=True,
      pipeline=DataPipelineConfig(
         steps={
            "scaler": {"name": "sklearn.preprocessing.StandardScaler"},
            "pca": {
               "name": "sklearn.decomposition.PCA",
               "n_components": 10,
            },
         }
      ),
   )
   data()
   print(f"X_train shape after pipeline: {data.X_train.shape}")

CLI example:

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data.dataset_name=make_classification \
      data.pipeline.steps.scaler.name=sklearn.preprocessing.StandardScaler \
      data.pipeline.steps.pca.name=sklearn.decomposition.PCA \
      data.pipeline.steps.pca.n_components=10

Sampling Strategies
~~~~~~~~~~~~~~~~~~~

Use pluggable samplers for robust train/test/validation splits:

.. code-block:: python

   from deckard.data import DataConfig
   from deckard.data.sample import KFoldSampler, ShuffleSampler

   # Standard 3-way split (default behavior with val_size)
   data = DataConfig(
      dataset_name="make_classification",
      data_params={"n_samples": 100, "n_features": 20, "random_state": 42},
      train_size=60,
      test_size=20,
      val_size=20,
      classifier=True,
   )

   # K-fold cross-validation with 5 folds
   data_kfold = DataConfig(
      dataset_name="make_classification",
      data_params={"n_samples": 100, "n_features": 20, "random_state": 42},
      classifier=True,
      sampler=KFoldSampler(n_splits=5, stratify=True),
   )

   # Repeated random splits (shuffle-split)
   data_shuffle = DataConfig(
      dataset_name="make_classification",
      data_params={"n_samples": 100, "n_features": 20, "random_state": 42},
      classifier=True,
      sampler=ShuffleSampler(n_splits=3, test_size=0.2, random_state=42),
   )

CLI examples:

.. code-block:: bash

   # Use k-fold sampling
   python -m deckard optimize --config-name experiment sample=kfold

   # Use shuffle split sampling
   python -m deckard optimize --config-name experiment sample=shuffle

   # Disable sampling (use no sampler)
   python -m deckard optimize --config-name experiment sample=none

DataConfig Sampling Examples (Repository Configs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

deckard includes ready-to-run sampling examples in
`examples/sklearn/config/data <../examples/sklearn/config/data>`_ and `examples/sklearn/config/sample <../examples/sklearn/config/sample>`_.

Key files:

- ``data/digits-kfold.yaml`` with ``sample: fold``
- ``data/digits-shuffle.yaml`` with ``sample: shuffle``
- ``data/digits-split.yaml`` with explicit ``val_size``
- ``sample/kfold.yaml`` defining :class:`deckard.data.sample.KFoldSampler`
- ``sample/shuffle.yaml`` defining :class:`deckard.data.sample.ShuffleSampler`

Example commands:

.. code-block:: bash

   # K-fold sampling path
   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name default \
      data=digits-kfold \
      sample=kfold

   # Shuffle-split sampling path
   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name default \
      data=digits-shuffle \
      sample=shuffle

   # Deterministic train/test/val split
   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name default \
      data=digits-split \
      sample=split

Fairlearn Data Support
~~~~~~~~~~~~~~~~~~~~~~

Fairlearn-aware data configs use
:class:`deckard.data.fairness.FairlearnDataConfig` with explicit
``sensitive_columns`` and optional fairness preprocessing defenses.

Repository examples:

- `examples/sklearn/config/data/fair-adult.yaml <../examples/sklearn/config/data/fair-adult.yaml>`_
- `examples/pytorch/config/data/fairlearn_celeba.yaml <../examples/pytorch/config/data/fairlearn_celeba.yaml>`_

The sklearn fair-adult example demonstrates correlation-remover preprocessing:

.. code-block:: yaml

   _target_: deckard.data.FairlearnDataConfig
   dataset_name: adult
   sensitive_columns: [sex]
   fairness_defense:
      step_name: fairness_correlation_remover
      name: fairlearn.preprocessing.CorrelationRemover

PyTorch Data Support
~~~~~~~~~~~~~~~~~~~~

PyTorch data workflows use :class:`deckard.data.pytorch.PytorchDataConfig`.

Repository examples:

- `examples/pytorch/config/data/torch_mnist.yaml <../examples/pytorch/config/data/torch_mnist.yaml>`_
- `examples/pytorch/config/data/torch_cifar10.yaml <../examples/pytorch/config/data/torch_cifar10.yaml>`_
- `examples/pytorch/config/data/fairlearn_celeba.yaml <../examples/pytorch/config/data/fairlearn_celeba.yaml>`_

Example command:

.. code-block:: bash

   python -m deckard optimize \
      --config-path examples/pytorch/config \
      --config-name torch_default \
      data=torch_mnist

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
