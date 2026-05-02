Model
============

The :mod:`~deckard.model` module defines the :class:`~deckard.model.ModelConfig` dataclass,
which provides a complete pipeline for **model configuration, training, evaluation, and persistence**.
It supports dynamic scikit-learn model instantiation, configurable parameters, CLI execution,
and integration with the :mod:`deckard.data` module.

.. automodule:: deckard.model
   :members:
   :show-inheritance:

Extensions
----------

Fairness Extension
~~~~~~~~~~~~~~~~~~

The fairness extension provides fairness-aware model behavior, including
group-sensitive fitting, scoring, and fairlearn defense wrappers.

.. automodule:: deckard.model.fairness
   :members:
   :show-inheritance:

Torch Extension
~~~~~~~~~~~~~~~

The torch extension provides PyTorch-native model training, prediction, and
scoring through a ``ModelConfig``-compatible API.

.. automodule:: deckard.model.pytorch
   :members:
   :show-inheritance:

Survival Extension
------------------

Survival-specific experiment orchestration is split into a dedicated optional
module.

.. automodule:: deckard.model.survival
   :members:
   :show-inheritance:

Overview
--------

:class:`~deckard.model.ModelConfig` automates the following steps:

* Dynamic instantiation of scikit-learn models via import strings (e.g. ``sklearn.svm.SVC``)
* Training, prediction, and evaluation for both classification and regression
* Timing instrumentation for training, prediction, and scoring
* Model persistence (save/load with ``pickle``)
* Hydra/YAML configuration for reproducibility and experiment management
* CLI support for one-line model training and testing

Supported frameworks
~~~~~~~~~~~~~~~~~~~~
Currently supports:
- **scikit-learn**

(Extendable to other frameworks in future versions.)

Usage
-----

Command-line example
~~~~~~~~~~~~~~~~~~~~

You can train and evaluate models directly from the terminal:

.. code-block:: bash

   # Integration-style logistic regression
   python -m deckard optimize --config-name experiment \
      model.model_type=sklearn.linear_model.LogisticRegression \
      model.model_params.max_iter=25

   # Integration-style random forest classifier
   python -m deckard optimize --config-name experiment \
      model.model_type=sklearn.ensemble.RandomForestClassifier \
      model.model_params.n_estimators=25 \
      model.model_params.random_state=42

   # Use a custom Hydra/YAML configuration
   python -m deckard optimize --config-path configs --config-name experiment


Programmatic example
~~~~~~~~~~~~~~~~~~~~

To use :class:`~deckard.model.ModelConfig` from Python:

.. code-block:: python

   from deckard.data import DataConfig
   from deckard.model import ModelConfig

   data = DataConfig(
      dataset_name="make_classification",
      data_params={
         "n_samples": 40,
         "n_features": 10,
         "n_informative": 4,
         "n_redundant": 0,
         "n_clusters_per_class": 1,
         "n_classes": 2,
         "random_state": 7,
      },
      train_size=30,
      test_size=10,
      random_state=42,
      stratify=True,
      classifier=True,
   )
   data()

   model = ModelConfig(
      model_type="sklearn.linear_model.LogisticRegression",
      classifier=True,
      model_params={"max_iter": 25},
   )

   scores = model(data)

   print(f"Scores: {scores}")

Regression example
~~~~~~~~~~~~~~~~~~

The integration suite also validates regression with the same API:

.. code-block:: python

   from deckard.data import DataConfig
   from deckard.model import ModelConfig

   reg_data = DataConfig(
      dataset_name="make_regression",
      data_params={
         "n_samples": 40,
         "n_features": 10,
         "n_informative": 5,
         "noise": 0.1,
         "random_state": 13,
      },
      train_size=30,
      test_size=10,
      random_state=42,
      classifier=False,
   )
   reg_data()

   reg_model = ModelConfig(
      model_type="sklearn.linear_model.LinearRegression",
      classifier=False,
   )
   reg_scores = reg_model(reg_data)
   print(reg_scores["mse"])

Custom configuration
~~~~~~~~~~~~~~~~~~~~

Example YAML configuration (``configs/model/rf.yaml``):

.. code-block:: yaml

   _target_: deckard.model.ModelConfig
   model_type: sklearn.linear_model.LogisticRegression
   classifier: True
   model_params:
      max_iter: 25

Internals
---------

Timing and logging
~~~~~~~~~~~~~~~~~~
All major operations (training, prediction, scoring, saving/loading) record wall-clock time
and log via Python’s ``logging`` module.

Scoring
~~~~~~~
* For classifiers: accuracy, precision, recall, and F1 score.
* For regressors: MSE, RMSE, and MAE.

Persistence
~~~~~~~~~~~
Models are saved and loaded using ``pickle`` via ``_save_model()`` and ``_load_model()``.

Troubleshooting
---------------

* **Model not fitted error** — train the model before calling ``_save_model`` or predictions.
* **Hydra config not found** — ensure the YAML file path is valid or use inline overrides.
* **pickle EOFError** — verify the model file is not corrupted.
* **CLI argument conflicts** — use ``conflict_handler='resolve'`` when composing parsers.
* **Probability prediction errors** — set ``--probability`` only for models that support ``predict_proba()``.


See also
~~~~~~~~
* :doc:`data`
* :doc:`attack`
* :doc:`experiment`
* :doc:`utils`
