Experiment
==========

The :mod:`deckard.experiment` module contains the high-level orchestration
entrypoints for end-to-end experiment execution.

.. automodule:: deckard.experiment
   :members:
   :show-inheritance:

Torch Extension
---------------

PyTorch-specific experiment orchestration is available via
:class:`deckard.experiment.torch_experiment.TorchExperimentConfig` in the
optional :mod:`deckard.experiment.torch_experiment` module.

Use this extension when you need PyTorch model/data orchestration while keeping
the same high-level experiment lifecycle as :class:`deckard.experiment.ExperimentConfig`.

Survival Extension
------------------

Survival-specific experiment orchestration is split into a dedicated optional
module.

.. automodule:: deckard.experiment.survival
   :members:
   :show-inheritance:

Overview
--------

The experiment layer coordinates the full Deckard workflow by composing:

- data loading and sampling via :mod:`deckard.data`
- model training/evaluation via :mod:`deckard.model`
- optional attack execution via :mod:`deckard.attack`
- score aggregation and file outputs via :mod:`deckard.file`

It is the primary integration point for reproducible end-to-end runs.

Available experiment entrypoints:

- :class:`deckard.experiment.ExperimentConfig` (default)
- :class:`deckard.experiment.torch_experiment.TorchExperimentConfig` (PyTorch)
- :class:`deckard.experiment.survival.SurvivalExperimentConfig` (survival)

Usage
-----

Command-line example
~~~~~~~~~~~~~~~~~~~~

Run an experiment from the project root:

.. code-block:: bash

   python -m deckard optimize --config-name experiment

   # With explicit model and data configuration
   python -m deckard optimize --config-name experiment \
      data.dataset_name=make_classification \
      data.data_params.n_samples=100 \
      model.model_type=sklearn.ensemble.RandomForestClassifier \
      model.model_params.n_estimators=50

   # With evasion attack and defense pipeline
   python -m deckard optimize --config-name experiment \
      attack.attack_type=art.attacks.evasion.FastGradientMethod \
      attack.attack_params.eps=0.1 \
      model.defense.defenses[0].defense_name=art.defences.preprocessor.FeatureSqueezing

   # PyTorch example config
   python -m deckard optimize \
      --config-path examples/pytorch/config \
      --config-name torch_default

   # Fairness-focused sklearn config
   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name fairness-default

Programmatic example
~~~~~~~~~~~~~~~~~~~~

Use the experiment config directly in Python with explicit configurations:

.. code-block:: python

   from deckard.attack import AttackConfig
   from deckard.data import DataConfig
   from deckard.experiment import ExperimentConfig
   from deckard.model import ModelConfig
   from deckard.model.defend import DefensePipelineConfig
   from deckard.score import DefaultClassifierConfig

   # Explicit data configuration
   data = DataConfig(
      dataset_name="make_classification",
      data_params={
         "n_samples": 100,
         "n_features": 20,
         "n_informative": 10,
         "n_redundant": 0,
         "n_clusters_per_class": 2,
         "n_classes": 2,
         "random_state": 42,
      },
      train_size=70,
      test_size=30,
      random_state=42,
      stratify=True,
      classifier=True,
      scorer=DefaultClassifierConfig(),
   )

   # Explicit model configuration
   model = ModelConfig(
      model_type="sklearn.ensemble.RandomForestClassifier",
      classifier=True,
      model_params={"n_estimators": 50, "max_depth": 10, "random_state": 42},
      scorer=DefaultClassifierConfig(),
   )

   # Optional defense configuration
   defense = DefensePipelineConfig(
      defenses=[
         {
            "defense_name": "art.defences.preprocessor.FeatureSqueezing",
            "defense_params": {"bit_depth": 8},
         }
      ]
   )
   model.defense = defense

   # Optional attack configuration
   attack = AttackConfig(
      attack_type="art.attacks.evasion.FastGradientMethod",
      attack_params={"eps": 0.15},
      attack_size=50,
   )

   # Compose the full experiment
   cfg = ExperimentConfig(data=data, model=model, attack=attack)
   scores = cfg()
   
   print("Experiment Results:")
   print(f"  Data shape: {data.X_train.shape}")
   print(f"  Model accuracy: {scores.get('accuracy', 'N/A')}")
   print(f"  Attack success: {scores.get('evasion_success_rate', 'N/A')}")

Internals
---------

The module resolves nested config objects, applies runtime overrides, and
normalizes outputs for downstream scoring/serialization.

Troubleshooting
---------------

- Verify config paths and override keys when Hydra/OmegaConf resolution fails.
- Ensure optional dependencies are installed for selected model/attack backends.
- Check file output paths in :class:`deckard.file.FileConfig` if artifacts are missing.

See also
~~~~~~~~

* :doc:`data` — data configuration and loading
* :doc:`model` — model configuration and training
* :doc:`attack` — attack configuration
* :doc:`file` — result serialization
* :doc:`score` — scoring framework
* :doc:`pytorch` — PyTorch experiment orchestration
* :doc:`lifelines` — survival experiment orchestration
* :doc:`utils` — utility functions
