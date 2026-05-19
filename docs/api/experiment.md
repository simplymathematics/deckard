# Experiment

The :mod:`deckard.experiment` module contains the high-level orchestration
entrypoints for end-to-end experiment execution.

.. automodule:: deckard.experiment
   :members:
   :show-inheritance:

## Torch Framework

PyTorch-specific experiment orchestration is available via
:class:`deckard.frameworks.pytorch.experiment.TorchExperimentConfig` in the
optional :mod:`deckard.frameworks.pytorch.experiment.TorchExperimentConfig` module.
See also: :doc:`pytorch`.

Use this extension when you need PyTorch model/data orchestration while keeping
the same high-level experiment lifecycle as :class:`deckard.experiment.ExperimentConfig`.

## Survival Plugin

Survival-specific experiment orchestration is split into a dedicated optional
module.
See also: :doc:`lifelines`.

.. automodule:: deckard.plugins.experiment.survival
   :members:
   :show-inheritance:

## Overview

The experiment layer coordinates the full deckard workflow by composing:

- data loading and sampling via :mod:`deckard.data`
- model training/evaluation via :mod:`deckard.model`
- optional attack execution via :mod:`deckard.attack`
- optional detector execution via :mod:`deckard.detector`
- score aggregation and file outputs via :mod:`deckard.file`

It is the primary integration point for reproducible end-to-end runs.

Available experiment entrypoints:

- :class:`~deckard.experiment.ExperimentConfig` (default)
- :class:`~deckard.frameworks.pytorch.experiment.TorchExperimentConfig` (PyTorch)
- :class:`~deckard.experiment.survival.SurvivalExperimentConfig` (survival)

## Examples

.. seealso::

   Notebook-based experiment workflows (single-attack, multi-attack,
   detector phase, and backend-specific runs) are documented in:

   - :doc:`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - :doc:`notebooks/art_attacks.ipynb </notebooks/art_attacks>`
   - :doc:`notebooks/pytorch.ipynb </notebooks/pytorch>`

## Internals

The module resolves nested config objects, applies runtime overrides, and
normalizes outputs for downstream scoring/serialization.

## Troubleshooting

- Verify config paths and override keys when Hydra/OmegaConf resolution fails.
- Ensure optional dependencies are installed for selected model/attack backends.
- Check file output paths in :class:`deckard.file.FileConfig` if artifacts are missing.

### See also

* :doc:`data` — data configuration and loading
* :doc:`model` — model configuration and training
* :doc:`attack` — attack configuration
* :doc:`file` — result serialization
* :doc:`score` — scoring framework
* :doc:`pytorch` — PyTorch experiment orchestration
* :doc:`lifelines` — survival experiment orchestration
* :doc:`utils` — utility functions
