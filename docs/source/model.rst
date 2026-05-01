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

   # Train and evaluate using defaults (Logistic Regression)
   python -m deckard optimize --config-name experiment model.model_type=sklearn.linear_model.LogisticRegression

   # Override model type and parameters
   python -m deckard optimize --config-name experiment \
      model.model_type=sklearn.ensemble.RandomForestClassifier \
      model.model_params.n_estimators=100 \
      model.model_params.max_depth=5

   # Use a custom Hydra/YAML configuration
   python -m deckard optimize --config-path configs --config-name experiment


Programmatic example
~~~~~~~~~~~~~~~~~~~~

To use :class:`~deckard.model.ModelConfig` from Python:

.. code-block:: python

   from deckard.data import initialize_data_config
   from deckard.model import initialize_model_config

   # Load data and initialize model
   data = data_config(dataset_name="adult", test_size=0.2, random_state=42)
   model = initialize_model_config(model_type="sklearn.ensemble.RandomForestClassifier", n_estimators=100, max_depth=5, random_state=42)

   # Call the data object to load/split the dataset
   data(data_filepath="data.pkl", score_file="data_scores.json")
   # Call the model object to train, predict, and score
   scores = model(data=data, model_filepath="models/rf.pkl", score_file="model_scores.json")

   print(f"Scores: {scores}")

Custom configuration
~~~~~~~~~~~~~~~~~~~~

Example YAML configuration (``configs/model/rf.yaml``):

.. code-block:: yaml

   _target_: model.ModelConfig
   model_type: sklearn.ensemble.RandomForestClassifier
   classifier: True
   probability: False
   model_params:
      n_estimators: 100
      max_depth: 5
      random_state: 42

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
