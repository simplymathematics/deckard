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

- **scikit-learn** — via :class:`~deckard.model.ModelConfig`
- **PyTorch** — via :class:`~deckard.model.pytorch.PytorchModelConfig`
- **Fairlearn (sklearn)** — via :class:`~deckard.model.fairness.FairlearnModelConfig`
- **Fairlearn (PyTorch)** — via :class:`~deckard.model.fairness.FairlearnPytorchModelConfig`

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

      The fairness extension provides fairness-aware model behavior, including
      group-sensitive fitting, scoring, and fairlearn defense wrappers.

      Two concrete classes are provided:

      - :class:`~deckard.model.fairness.FairlearnModelConfig` — for sklearn models;
         inherits :class:`~deckard.model.ModelConfig` and adds fairness-aware scoring
         and fairlearn defense support.
      - :class:`~deckard.model.fairness.FairlearnPytorchModelConfig` — for PyTorch
         models; inherits :class:`~deckard.model.pytorch.PytorchModelConfig` directly
         so all torch training, device management, ART wrapping, and checkpointing
         comes for free, with fairness-aware scoring layered on top via
         ``_FairnessBehaviorMixin``.

      .. automodule:: deckard.model.fairness
          :members:
          :show-inheritance:
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

Defense Pipelines
~~~~~~~~~~~~~~~~~

Apply adversarial robustness defenses using :class:`~deckard.model.DefensePipelineConfig` 
to chain multiple ART-based defenses:

.. code-block:: python

   from deckard.data import DataConfig
   from deckard.model import ModelConfig
   from deckard.model.defend import DefensePipelineConfig

   # Train a baseline model
   data = DataConfig(
      dataset_name="make_classification",
      data_params={
         "n_samples": 60,
         "n_features": 10,
         "n_informative": 5,
         "random_state": 42,
      },
      train_size=40,
      test_size=20,
      classifier=True,
   )
   data()

   model = ModelConfig(
      model_type="sklearn.linear_model.LogisticRegression",
      classifier=True,
      model_params={"max_iter": 50},
   )
   model(data)

   # Apply a defense pipeline with feature squeezing preprocessor and gaussian noise postprocessor
   defense_pipeline = DefensePipelineConfig(
      defenses=[
         {
            "defense_name": "art.defences.preprocessor.FeatureSqueezing",
            "defense_params": {"bit_depth": 8},
         },
         {
            "defense_name": "art.defences.postprocessor.GaussianNoise",
            "defense_params": {"sigma": 0.1},
         },
      ]
   )

   # Apply the defense to the trained model
   defended_estimator = defense_pipeline.apply_to_trained_model(data=data, model=model)
   print(f"Defense applied in {defense_pipeline.defense_application_time:.4f}s")

CLI example:

.. code-block:: bash

   # Apply defenses via YAML config
   python -m deckard optimize --config-name experiment \
      model.defense.defenses[0].defense_name=art.defences.preprocessor.FeatureSqueezing \
      model.defense.defenses[0].defense_params.bit_depth=8 \
      model.defense.defenses[1].defense_name=art.defences.postprocessor.GaussianNoise \
      model.defense.defenses[1].defense_params.sigma=0.1

Example YAML configuration (``configs/model/defended.yaml``):

.. code-block:: yaml

   _target_: deckard.model.ModelConfig
   model_type: sklearn.linear_model.LogisticRegression
   classifier: True
   model_params:
      max_iter: 50
   defense:
      _target_: deckard.model.DefensePipelineConfig
      defenses:
         - defense_name: art.defences.preprocessor.FeatureSqueezing
           defense_params:
              bit_depth: 8
         - defense_name: art.defences.postprocessor.GaussianNoise
           defense_params:
              sigma: 0.1

Defense Chains in Config Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can build reproducible defense chains by composing entries from
``examples/sklearn/config/defense`` (for scikit-learn workflows) or
``examples/pytorch/config/defense`` (for PyTorch workflows).

For example, the following chain applies an ART preprocessor first and then an
ART postprocessor:

.. code-block:: yaml

   defense:
      _target_: deckard.model.DefensePipelineConfig
      defenses:
         - defense_name: art.defences.preprocessor.FeatureSqueezing
           defense_params:
              bit_depth: 8
              clip_values: [0, 255]
         - defense_name: art.defences.postprocessor.GaussianNoise
           defense_params:
              scale: 0.2
              apply_predict: true

Fairlearn defenses are also supported in pipeline form, including wrappers like
``fairlearn.reductions.ExponentiatedGradient`` and
``fairlearn.postprocessing.ThresholdOptimizer``.

PyTorch Support Examples
~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch model workflows are configured through
``examples/pytorch/config/torch_default.yaml`` with model settings in
``examples/pytorch/config/model/default.yaml``.

Example command:

.. code-block:: bash

   python -m deckard optimize \
      --config-path examples/pytorch/config \
      --config-name torch_default

This uses :class:`deckard.model.pytorch.PytorchModelConfig` and supports:

- configurable optimizer/criterion
- ART-compatible wrapping for attack evaluation
- optional fairness defenses (for example,
  ``examples/pytorch/config/defense/fairlearn-adversarial-classifier.yaml``)

Fairlearn Model Support
~~~~~~~~~~~~~~~~~~~~~~~

Deckard's fairness model extension supports fairlearn-backed defenses and model
wrappers in both sklearn and PyTorch-centered workflows.

Common fairlearn defense chain usage:

.. code-block:: bash

   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name fairness-default \
      defense=fairlearn-exponentiated-gradient \
      score=fairness-classification

Legacy Single-Defense Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~deckard.model.DefensePipelineConfig` automatically converts legacy single-defense YAML configurations:

.. code-block:: yaml

   # Legacy format (automatically converted to pipeline)
   _target_: deckard.model.DefenseConfig
   defense_name: art.defences.preprocessor.FeatureSqueezing
   defense_params:
      bit_depth: 8

   # Is converted to
   _target_: deckard.model.DefensePipelineConfig
   defenses:
      - defense_name: art.defences.preprocessor.FeatureSqueezing
        defense_params:
           bit_depth: 8

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
Use the public model persistence interfaces:

- ``model.save(filepath)``
- ``model.load(filepath)``
- ``model(data, model_file=...)`` for automatic load-or-train behavior

For scikit-learn-backed :class:`~deckard.model.ModelConfig`, persisted models
use the framework's object serialization path via the config base save/load
machinery.

For :class:`~deckard.model.pytorch.PytorchModelConfig`, persistence is explicit
and torch-native:

- ``save`` writes a checkpoint payload with model metadata plus
  ``state_dict`` using ``torch.save``.
- ``load`` restores metadata and calls ``load_state_dict``.

Public API example (automatic load-or-train):

.. code-block:: python

   from deckard.model import ModelConfig

   model = ModelConfig(
      model_type="sklearn.linear_model.LogisticRegression",
      classifier=True,
   )
   # If model_file exists, it is loaded. Otherwise, the model is trained and saved.
   scores = model(data, model_file="outputs/models/logreg.pkl")

Public API example (PyTorch save/load):

.. code-block:: python

   from deckard.model.pytorch import PytorchModelConfig

   model = PytorchModelConfig(
      model_type="torch_example.py:ResNet18",
      model_params={"num_channels": 1, "num_classes": 10},
      classifier=True,
   )
   model(data)
   model.save("outputs/models/torch_resnet18.pt")

   reloaded = PytorchModelConfig(
      model_type="torch_example.py:ResNet18",
      model_params={"num_channels": 1, "num_classes": 10},
      classifier=True,
   )
   reloaded.load("outputs/models/torch_resnet18.pt")

Pre-trained torch models
^^^^^^^^^^^^^^^^^^^^^^^^

There are two supported patterns:

1. Load a previously saved Deckard PyTorch checkpoint via ``load(filepath)``.
2. Point ``model_type`` to a custom constructor/class that returns an already
   initialized ``nn.Module`` (for example, one that internally loads external
   pre-trained weights), then run normal Deckard training/evaluation.

If you want inference-only behavior from a pre-trained checkpoint, load it via
``load`` and then call the model with ``model_file``/prediction outputs as
needed, without requiring private methods.

Troubleshooting
---------------

* **Model not fitted error** — train the model before calling ``_save_model`` or predictions.
* **Hydra config not found** — ensure the YAML file path is valid or use inline overrides.
* **pickle EOFError** — verify the model file is not corrupted.
* **CLI argument conflicts** — use ``conflict_handler='resolve'`` when composing parsers.
* **Probability prediction errors** — set ``--probability`` only for models that support ``predict_proba()``.


See also
~~~~~~~~
* :doc:`data` — data configuration and loading
* :doc:`experiment` — experiment orchestration
* :doc:`attack` — attack configuration
* :doc:`score` — scoring framework
* :doc:`pytorch` — PyTorch model integration
* :doc:`anjana` — anonymization-aware models
* :doc:`lifelines` — survival model configuration
* :doc:`utils` — utility functions
