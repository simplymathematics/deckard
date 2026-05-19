# Model

The {mod}`~deckard.model` module defines the {class}`~deckard.model.ModelConfig` dataclass,
which provides a complete pipeline for **model configuration, training, evaluation, and persistence**.
It supports dynamic scikit-learn model instantiation, configurable parameters, CLI execution,
and integration with the {mod}`deckard.data` module.

```{eval-rst}
.. automodule:: deckard.model
   :members:
   :show-inheritance:
```


## Extensions

### Fairlearn Extension

The fairlearn extension provides fairness-aware model behavior, including
group-sensitive fitting, scoring, and fairlearn defense wrappers.
See also: {doc}`fairlearn`.

```{eval-rst}
.. automodule:: deckard.plugins.fairlearn.model
   :members:
   :show-inheritance:
```


### Pytorch extesion

The Pytorch extesion provides PyTorch-native model training, prediction, and
scoring through a {class}`~deckard.model.ModelConfig`-compatible API.
See also: {doc}`pytorch`.

```{eval-rst}
.. automodule:: deckard.frameworks.pytorch.model
   :members:
   :show-inheritance:
```


## Lifelines plugin

Survival-specific experiment orchestration is split into a dedicated optional
module.
See also: {doc}`lifelines`.

```{eval-rst}
.. automodule:: deckard.plugins.lifelines.model
   :members:
   :show-inheritance:
```


## Overview

{class}`~deckard.model.ModelConfig` automates the following steps:

* Dynamic instantiation of scikit-learn models via import strings (e.g. ``sklearn.svm.SVC``)
* Training, prediction, and evaluation for both classification and regression
* Timing instrumentation for training, prediction, and scoring
* Model persistence (save/load with ``pickle``)
* Hydra/YAML configuration for reproducibility and experiment management
* CLI support for one-line model training and testing

### Model scoring mode

{class}`~deckard.model.ModelConfig` supports split-aware scoring with
``score_mode`` set to one of:

- ``train``
- ``test``
- ``val``

The experiment layer can propagate this mode automatically so model scoring is
performed on the active split selected by experiment scoring policy.

### Supported frameworks
Currently supports:

- **scikit-learn** — via {class}`~deckard.model.ModelConfig`
- **PyTorch** — via {class}`~deckard.frameworks.pytorch.model.PytorchModelConfig`
- **Fairlearn (sklearn)** — via {class}`~deckard.plugins.fairlearn.model.FairlearnModelConfig`
- **Fairlearn (PyTorch)** — via {class}`~deckard.plugins.fairlearn.model.FairlearnPytorchModelConfig`

## Examples

```{seealso}

   Notebook-based model workflows, including sklearn, defense pipelines,
   Fairlearn, and PyTorch models, are documented in:

   - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - {doc}`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`
   - {doc}`notebooks/art_defenses.ipynb </notebooks/art_defenses>`

```
## Internals

### Timing and logging
All major operations (training, prediction, scoring, saving/loading) record wall-clock time
and log via Python’s ``logging`` module.

### Scoring
* For classifiers: accuracy, precision, recall, and F1 score.
* For regressors: MSE, RMSE, and MAE.

### Persistence
Use the public model persistence interfaces:

- :meth:`deckard.model.ModelConfig.save`
- :meth:`deckard.model.ModelConfig.load`
- ``model(data, model_file=...)`` for automatic load-or-train behavior

For scikit-learn-backed {class}`~deckard.model.ModelConfig`, persisted models
use the framework's object serialization path via the config base save/load
machinery.

For {class}`~deckard.frameworks.pytorch.model.PytorchModelConfig`, persistence is explicit
and torch-native:

- ``save`` writes a checkpoint payload with model metadata plus
  ``state_dict`` using ``torch.save``.
- ``load`` restores metadata and calls ``load_state_dict``.

Public API example (automatic load-or-train): see the {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`
notebook for executed save/load examples.

Public API example (PyTorch save/load): see the {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`
notebook for executed PyTorch checkpoint save/load patterns.

#### Pre-trained torch models

There are two supported patterns:

1. Load a previously saved deckard PyTorch checkpoint via ``load(filepath)``.
2. Point ``model_type`` to a custom constructor/class that returns an already
   initialized {class}`torch.nn.Module` (for example, one that internally loads external
   pre-trained weights), then run normal deckard training/evaluation.

If you want inference-only behavior from a pre-trained checkpoint, load it via
``load`` and then call the model with ``model_file``/prediction outputs as
needed, without requiring private methods.

## Troubleshooting

* **Model not fitted error** — train the model before calling
   :meth:`deckard.model.ModelConfig.save` or predictions.
* **Hydra config not found** — ensure the YAML file path is valid or use inline overrides.
* **pickle EOFError** — verify the model file is not corrupted.
* **CLI argument conflicts** — use ``conflict_handler='resolve'`` when composing parsers.
* **Probability prediction errors** — set ``--probability`` only for models that support ``predict_proba()``.


### See also
* {doc}`data` — data configuration and loading
* {doc}`experiment` — experiment orchestration
* {doc}`attack` — attack configuration
* {doc}`score` — scoring framework
* {doc}`pytorch` — PyTorch model integration
* {doc}`anjana` — anonymization-aware models
* {doc}`lifelines` — survival model configuration
* {doc}`utils` — utility functions
