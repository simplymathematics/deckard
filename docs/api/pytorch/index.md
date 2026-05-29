# PyTorch Integration

deckard provides native support for PyTorch models, data, and experiments through
the optional Pypytorch extesion modules. This integration enables seamless use of
PyTorch-based workflows within the deckard framework.

The PyTorch runtime still follows the canonical Deckard model contract,
including stage-aware defense application, files-only persistence, and
canonical timing and score fields.

## Parent Core Modules and Behavior Deltas

Parent core pages:

- {doc}`../data/index`
- {doc}`../model/index`
- {doc}`../experiment/index`

Behavior deltas in this integration:

- torch-native data/model classes and device reconciliation,
- ART estimator wrapping for torch model attack/defense compatibility,
- torch runtime artifact handling layered over canonical file/timing contracts.

(pytorch-overview)=

## Overview

The PyTorch integration consists of three main extension modules:

- {mod}`deckard.frameworks.pytorch.data` — PyTorch dataset and DataLoader configuration
- {mod}`deckard.frameworks.pytorch.model` — PyTorch model training and evaluation
- {mod}`deckard.frameworks.pytorch.experiment` — end-to-end PyTorch experiment orchestration

These modules are fully integrated with deckard's attack, defense, and scoring
pipelines, allowing adversarial robustness studies on PyTorch models.

### Key Features

- **Device reconciliation**: automatic CPU/CUDA/MPS device selection and validation
- **ART integration**: PyTorch models wrap as ART estimators for attack/defense
- **Fairness support**: compatible with {mod}`deckard.plugins.fairlearn.data`
  and attack
- **Fairness support**:
  {class}`~deckard.plugins.fairlearn.model.FairlearnPytorchModelConfig`
inherits {class}`~deckard.frameworks.pytorch.model.PytorchModelConfig` directly
and adds
  fairness-aware scoring; compatible with {mod}`deckard.plugins.fairlearn.data` and
  attack stratification by sensitive features
- The shared fairness model mixin is {class}`~deckard.plugins.fairlearn.model.FairnessBehaviorMixin`.
- **Survival analysis**: optional integration with lifelines-based survival experiments
- **Standard scorers**: classification, regression, and attack metrics via
  {class}`deckard.score.DefaultClassifierScorerDictConfig`, etc.

### Data Loading

The {class}`~deckard.frameworks.pytorch.data.PytorchDataConfig` extends {class}`deckard.data.DataConfig`
with PyTorch-specific behavior:

- Wraps datasets as {class}`torch.utils.data.Dataset` instances
- Provides configurable {class}`torch.utils.data.DataLoader` for batching
- Supports device placement for GPU-accelerated data loading
- Integrates with {mod}`deckard.plugins.fairlearn.data` for stratified sampling

Dataset discovery and naming notes for torch-backed workflows:

- torchvision discovery registers canonical names as `torchvision.<DatasetClass>`
  (for example `torchvision.MNIST`)
- torchvision compatibility aliases include `torchvision_<DatasetClass>` and
  `torchvision.datasets.<DatasetClass>`
- fairlearn local dataset declarations are exposed as
  `fairlearn.TinyFairness` and `fairlearn.SyntheticImageSensitiveDataset`
  (with compatibility aliases for the fully-qualified deckard class names)

Common torch transform and data pipeline components:

- [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [`torchvision.transforms.Compose`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html)
- [`torchvision.transforms.Normalize`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html)

### Model Configuration

The {class}`~deckard.frameworks.pytorch.model.PytorchModelConfig` supports:

- Any PyTorch {class}`torch.nn.Module` via import path specification
- Configurable optimizers (SGD, Adam, AdamW, etc.)
- Learning rate scheduling via PyTorch LR schedulers
- Device handling with automatic precision selection (float32/float64)
- Optional early stopping and checkpoint management
- Integration with ART's {class}`~art.estimators.classification.PyTorchClassifier`
  and {class}`~art.estimators.regression.PyTorchRegressor`

Common ART attack and defense components used with torch workflows:

- [`FastGradientMethod`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm)
- [`ProjectedGradientDescent`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#projected-gradient-descent-pgd)
- [`FeatureSqueezing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#feature-squeezing)
- [`AdversarialTrainer`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/trainer.html#adversarial-training)

See also {doc}`../attack/index`, {doc}`../model/index`, and {doc}`../score/index` for cross-component
composition.

### Persistence Contract

- {meth}`deckard.frameworks.pytorch.model.PytorchModelConfig.save`/{meth}`deckard.frameworks.pytorch.model.PytorchModelConfig.load` on {class}`~deckard.frameworks.pytorch.model.PytorchModelConfig`
  persist config state as YAML.
- {meth}`deckard.frameworks.pytorch.model.PytorchModelConfig.save_model`/{meth}`deckard.frameworks.pytorch.model.PytorchModelConfig.load_model` persist runtime torch model state artifacts.
- Runtime torch artifacts use `.pt` (and optionally pickle-compatible payloads
  where supported by the runtime loader).
- During checkpointing, YAML config records include references to runtime
  `model_state_file` entries.

### Canon Runtime Contract

PyTorch data configs participate in the same canonical data runtime contract as
core and plugin families:

- files-only persistence through `files={...}` aliases
- canonical timing keys in `times`
- split-scoped score mode (`train|test|val|all`)
- stage lifecycle hook orchestration owned by the core data runtime
- model defenses follow the same stage-aware contract documented in {doc}`../model/index`

See {doc}`../data/index` and {doc}`../data/pipeline` for canonical stage/scope semantics.

### Experiment Orchestration

The {class}`~deckard.frameworks.pytorch.experiment.TorchExperimentConfig` enforces:

- All data/model/attack components use PyTorch backend
- Unified device selection across all components
- Automatic device reconciliation to prevent device mismatch errors
- Lifecycle management (training, evaluation, attack, scoring)

## Examples

```{seealso}

  Notebook-based PyTorch workflows (training, attacks, defenses, and
  fairness-integrated evaluation) are documented in:

  - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`
  - {doc}`notebooks/fairlearn.ipynb </notebooks/fairlearn>`

```

### Troubleshooting

- **Device mismatch errors**: Verify all components use compatible devices. The
  {class}`~deckard.frameworks.pytorch.experiment.TorchExperimentConfig` will raise
  an error if conflicts are detected.
- **Out of memory (OOM)**: Reduce batch_size, model size, or use gradient
  checkpointing. Consider using mixed precision training.
- **Missing PyTorch modules**: Ensure torchvision is installed for common models
  like ResNet. Install via `pip install "deckard[pytorch]"` or similar.
- **ART compatibility**: Use ART-supported model architectures. Custom modules may
  need additional ART estimator wrapping.
- **Artifact extension mismatch**: Use YAML for config objects and `.pt` for
  runtime PyTorch model-state artifacts.

### See also

- {doc}`../data/index` — general data configuration including {mod}`deckard.frameworks.pytorch.data`
- {doc}`../model/index` — general model configuration including {mod}`deckard.frameworks.pytorch.model`
- {doc}`../experiment/index` — experiment orchestration including {class}`deckard.frameworks.pytorch.experiment.TorchExperimentConfig`
- {doc}`../attack/index` — attack configuration and execution
- {doc}`../plot/index` — visualization support including training history plots
- {doc}`../../overview/extensions/lifelines` — optional survival analysis integration with PyTorch
- {doc}`../../developers/data/data` — cross-family runtime contract
- {doc}`../../developers/contributor/migration` — migration guardrails
