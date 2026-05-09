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

The experiment layer coordinates the full deckard workflow by composing:

- data loading and sampling via :mod:`deckard.data`
- model training/evaluation via :mod:`deckard.model`
- optional attack execution via :mod:`deckard.attack`
- optional detector execution via :mod:`deckard.detector`
- score aggregation and file outputs via :mod:`deckard.file`

It is the primary integration point for reproducible end-to-end runs.

Available experiment entrypoints:

- :class:`deckard.experiment.ExperimentConfig` (default)
- :class:`deckard.experiment.torch_experiment.TorchExperimentConfig` (PyTorch)
- :class:`deckard.experiment.survival.SurvivalExperimentConfig` (survival)

Usage
-----

Multi-Attack Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

``ExperimentConfig.attack`` accepts either a single attack spec or a list of
attack specs.

- Single attack: unchanged behavior.
- Multi-attack: each configured attack must set a unique, non-empty ``alias``.
- Score keys: only colliding keys are suffixed as ``_<alias>``.
- Detector: receives pooled attack samples from all configured attacks.

Scoring Mode Policy
~~~~~~~~~~~~~~~~~~~

``ExperimentConfig`` supports split-aware scoring orchestration through:

- ``evaluation_mode``: high-level policy (``standard``, ``tuning``, ``report``)
- ``score_mode``: one explicit mode
- ``score_modes``: ordered list of modes to evaluate

Supported score modes are:

- ``pre-sample``
- ``train``
- ``test``
- ``val``

If no explicit score mode is set, defaults are:

- ``evaluation_mode=tuning`` -> ``test``
- ``evaluation_mode=report`` -> ``val``
- otherwise -> ``test``

Mode permutations by scorer type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``ExperimentConfig.score`` is a data-profile scorer, all supported modes
can be evaluated, including ``pre-sample``. Mode inputs are routed directly
from data splits/full dataset.

When ``ExperimentConfig.score`` is not a data-profile scorer (for example,
model prediction metrics), ``pre-sample`` is rejected because that mode is
reserved for full-dataset diagnostics.

Mode-specific output key normalization:

- ``val`` metrics are prefixed with ``validation_``
- ``pre-sample`` metrics are prefixed with ``presample_``

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

   # Multi-attack using list syntax on the same attack field
   python -m deckard optimize --config-name experiment \
      '+attack=[{"attack_type":"art.attacks.evasion.FastGradientMethod","attack_params":{"eps":0.05},"attack_size":20,"alias":"fgm"},{"attack_type":"art.attacks.evasion.HopSkipJump","attack_params":{"max_iter":5},"attack_size":20,"alias":"hsj"}]'

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

.. seealso::

   Fully-executed programmatic examples — including single-attack, multi-attack,
   defense pipelines, and detector phase — are available in the
   :doc:`notebooks/sklearn.ipynb </notebooks/sklearn>` and :doc:`notebooks/art_attacks.ipynb </notebooks/art_attacks>` notebooks.

Multi-attack programmatic example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the :doc:`notebooks/art_attacks.ipynb </notebooks/art_attacks>` notebook for a multi-attack experiment
with ``HopSkipJump`` alongside ``FastGradientMethod``.

Detector phase
~~~~~~~~~~~~~~

Detector configs can be attached to ``ExperimentConfig.detector`` to train and
evaluate an auxiliary detector after attack generation. See the
:doc:`notebooks/detector.ipynb </notebooks/detector>` notebook for an executed example.

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
