PyTorch Integration
===================

deckard provides native support for PyTorch models, data, and experiments through
the optional PyTorch extension modules. This integration enables seamless use of
PyTorch-based workflows within the deckard framework.

.. _pytorch-overview:

Overview
--------

The PyTorch integration consists of three main extension modules:

- :mod:`deckard.data.pytorch` — PyTorch dataset and DataLoader configuration
- :mod:`deckard.model.pytorch` — PyTorch model training and evaluation
- :mod:`deckard.experiment.torch_experiment` — end-to-end PyTorch experiment orchestration

These modules are fully integrated with deckard's attack, defense, and scoring
pipelines, allowing adversarial robustness studies on PyTorch models.

Key Features
~~~~~~~~~~~~

- **Device reconciliation**: automatic CPU/CUDA/MPS device selection and validation
- **ART integration**: PyTorch models wrap as ART estimators for attack/defense
- **Fairness support**: compatible with :mod:`deckard.data.fairness` and attack
- **Fairness support**: :class:`~deckard.model.fairness.FairlearnPytorchModelConfig`
  inherits :class:`~deckard.model.pytorch.PytorchModelConfig` directly and adds
  fairness-aware scoring; compatible with :mod:`deckard.data.fairness` and
  attack stratification by sensitive features
- **Survival analysis**: optional integration with lifelines-based survival experiments
- **Standard scorers**: classification, regression, and attack metrics via
  :class:`deckard.score.DefaultClassifierConfig`, etc.

Data Loading
~~~~~~~~~~~~

The :class:`~deckard.data.pytorch.PytorchDataConfig` extends :class:`deckard.data.DataConfig`
with PyTorch-specific behavior:

- Wraps datasets as :class:`torch.utils.data.Dataset` instances
- Provides configurable :class:`torch.utils.data.DataLoader` for batching
- Supports device placement for GPU-accelerated data loading
- Integrates with :mod:`deckard.data.fairness` for stratified sampling

Model Configuration
~~~~~~~~~~~~~~~~~~~

The :class:`~deckard.model.pytorch.PytorchModelConfig` supports:

- Any PyTorch :class:`torch.nn.Module` via import path specification
- Configurable optimizers (SGD, Adam, AdamW, etc.)
- Learning rate scheduling via PyTorch LR schedulers
- Device handling with automatic precision selection (float32/float64)
- Optional early stopping and checkpoint management
- Integration with ART's :class:`~art.estimators.classification.PyTorchClassifier`
  and :class:`~art.estimators.regression.PyTorchRegressor`

Experiment Orchestration
~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~deckard.experiment.torch_experiment.TorchExperimentConfig` enforces:

- All data/model/attack components use PyTorch backend
- Unified device selection across all components
- Automatic device reconciliation to prevent device mismatch errors
- Lifecycle management (training, evaluation, attack, scoring)

Examples
--------

.. seealso::

  Notebook-based PyTorch workflows (training, attacks, defenses, and
  fairness-integrated evaluation) are documented in:

  - :doc:`notebooks/pytorch.ipynb </notebooks/pytorch>`
  - :doc:`notebooks/fairlearn.ipynb </notebooks/fairlearn>`

Troubleshooting
~~~~~~~~~~~~~~~

- **Device mismatch errors**: Verify all components use compatible devices. The
  :class:`~deckard.experiment.torch_experiment.TorchExperimentConfig` will raise
  an error if conflicts are detected.
- **Out of memory (OOM)**: Reduce batch_size, model size, or use gradient
  checkpointing. Consider using mixed precision training.
- **Missing PyTorch modules**: Ensure torchvision is installed for common models
  like ResNet. Install via ``pip install "deckard[pytorch]"`` or similar.
- **ART compatibility**: Use ART-supported model architectures. Custom modules may
  need additional ART estimator wrapping.

See also
~~~~~~~~

* :doc:`data` — general data configuration including :mod:`deckard.data.pytorch`
* :doc:`model` — general model configuration including :mod:`deckard.model.pytorch`
* :doc:`experiment` — experiment orchestration including :class:`TorchExperimentConfig`
* :doc:`attack` — attack configuration and execution
* :doc:`plot` — visualization support including training history plots
* :doc:`lifelines` — optional survival analysis integration with PyTorch
* :doc:`modules` — overview of all extensions
