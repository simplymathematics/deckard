PyTorch Integration
===================

Deckard provides native support for PyTorch models, data, and experiments through
the optional PyTorch extension modules. This integration enables seamless use of
PyTorch-based workflows within the Deckard framework.

.. _pytorch-overview:

Overview
--------

The PyTorch integration consists of three main extension modules:

- :mod:`deckard.data.pytorch` — PyTorch dataset and DataLoader configuration
- :mod:`deckard.model.pytorch` — PyTorch model training and evaluation
- :mod:`deckard.experiment.torch_experiment` — end-to-end PyTorch experiment orchestration

These modules are fully integrated with Deckard's attack, defense, and scoring
pipelines, allowing adversarial robustness studies on PyTorch models.

Key Features
~~~~~~~~~~~~

- **Device reconciliation**: automatic CPU/CUDA/MPS device selection and validation
- **ART integration**: PyTorch models wrap as ART estimators for attack/defense
- **Fairness support**: compatible with :mod:`deckard.data.fairness` and attack
  stratification by sensitive features
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

Usage
-----

Command-line examples
~~~~~~~~~~~~~~~~~~~~~

**Basic PyTorch experiment with default config:**

.. code-block:: bash

   python -m deckard optimize \
      --config-path examples/pytorch/config \
      --config-name torch_default

**PyTorch with custom model and data:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=pytorch \
      data.dataset_name=CIFAR10 \
      model=pytorch \
      model.model_type=torchvision.models.resnet18

**PyTorch with evasion attack and defense:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=pytorch \
      model=pytorch \
      attack.attack_type=art.attacks.evasion.FastGradientMethod \
      attack.attack_params.eps=0.1 \
      model.defense.defenses[0].defense_name=art.defences.preprocessor.FeatureSqueezing

**PyTorch with fairness-aware attack metrics:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=fairness \
      data.base_data_config=pytorch \
      data.sensitive_feature=gender \
      model=pytorch \
      attack=fairlearn-attack \
      score.attack=fairlearn-attack

Programmatic examples
~~~~~~~~~~~~~~~~~~~~~

**Basic PyTorch experiment:**

.. code-block:: python

   from deckard.data.pytorch import PytorchDataConfig
   from deckard.model.pytorch import PytorchModelConfig
   from deckard.experiment.torch_experiment import TorchExperimentConfig
   from deckard.score import DefaultClassifierConfig
   import torch
   import torch.nn as nn

   # Define a simple PyTorch model
   class SimpleCNN(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
           self.fc = nn.Linear(32 * 32 * 32, num_classes)

       def forward(self, x):
           x = torch.relu(self.conv1(x))
           x = x.view(x.size(0), -1)
           return self.fc(x)

   # Configure data
   data = PytorchDataConfig(
       dataset_name="CIFAR10",
       train_size=45000,
       test_size=5000,
       val_size=5000,
       device="auto",
       classifier=True,
       scorer=DefaultClassifierConfig(),
   )

   # Configure model
   model = PytorchModelConfig(
       model_type="__main__.SimpleCNN",
       classifier=True,
       device="auto",
       optimizer_type="torch.optim.Adam",
       optimizer_params={"lr": 0.001},
       epochs=10,
       batch_size=32,
       scorer=DefaultClassifierConfig(),
   )

   # Run experiment
   cfg = TorchExperimentConfig(data=data, model=model)
   scores = cfg()
   print("Accuracy:", scores.get("accuracy"))

**PyTorch with attacks and fairness:**

.. code-block:: python

   from deckard.attack import AttackConfig
   from deckard.data.fairness import FairlearnDataConfig
   from deckard.score.attack import FairlearnAttackScorerConfig

   # Fairness-aware data with PyTorch backend
   data = FairlearnDataConfig(
       base_data_config=PytorchDataConfig(
           dataset_name="CIFAR10",
           train_size=40000,
           test_size=5000,
           val_size=5000,
           device="auto",
           classifier=True,
           scorer=DefaultClassifierConfig(),
       ),
       sensitive_feature="gender",  # attribute column name
   )

   # Attack configuration
   attack = AttackConfig(
       attack_type="art.attacks.evasion.FastGradientMethod",
       attack_params={"eps": 0.1},
   )

   # Fairness-stratified attack scoring
   attack_scorer = FairlearnAttackScorerConfig(
       attack_kind="evasion",
       scorers={"success_rate": "sklearn.metrics.accuracy_score"},
   )

   # Orchestrate
   cfg = TorchExperimentConfig(
       data=data,
       model=model,
       attack=attack,
   )
   scores = cfg()

Configuration
~~~~~~~~~~~~~

Key configuration options for :class:`~deckard.model.pytorch.PytorchModelConfig`:

- **model_type** (str): fully qualified import path to :class:`torch.nn.Module`
- **device** (str): "cpu", "cuda", "mps", or "auto" for automatic selection
- **optimizer_type** (str): e.g., "torch.optim.Adam", "torch.optim.SGD"
- **optimizer_params** (dict): keyword arguments for optimizer constructor
- **loss_fn_type** (str): e.g., "torch.nn.CrossEntropyLoss"
- **epochs** (int): number of training epochs
- **batch_size** (int): batch size for training/evaluation
- **scheduler_type** (str, optional): e.g., "torch.optim.lr_scheduler.StepLR"
- **early_stopping_patience** (int, optional): epochs to wait before stopping if
  validation loss plateaus

Device Management
~~~~~~~~~~~~~~~~~

The :class:`~deckard.experiment.torch_experiment.TorchExperimentConfig` handles
device reconciliation automatically. If you specify different devices in data,
model, or attack configs, it will:

1. Collect all specified devices
2. Select the most specific device (cuda > mps > cpu)
3. Validate no conflicts exist
4. Apply the unified device to all components

Example device specifications:

.. code-block:: python

   cfg = TorchExperimentConfig(
       data=data,  # device="cuda:0"
       model=model,  # device="auto"
       attack=attack,  # device=None
   )
   # Result: all components use device="cuda:0"

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
* :doc:`package` — overview of all extensions
