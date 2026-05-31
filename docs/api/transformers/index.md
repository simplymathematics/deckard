# Transformers Integration

Deckard provides transformer-native model support through
{mod}`deckard.frameworks.transformers` while preserving the canonical runtime
contracts for data, attack, scoring, and persistence.

The transformers runtime follows the same core Deckard model contract,
including stage-aware defense application, files-only persistence, and
canonical timing and score fields.

## Parent Core Modules and Behavior Deltas

Parent core pages:

- {doc}`../data/index`
- {doc}`../model/index`
- {doc}`../experiment/index`

Behavior deltas in this integration:

- tokenizer-aware wrapper declarations for text-classification workflows,
- transformer-specific ART model adaptation for robustness evaluation,
- compatibility layering over torch-backed dataset and scoring runtime paths.

(transformers-overview)=

## Overview

The transformers integration centers on two extension modules:

- {mod}`deckard.frameworks.transformers.declarations` — transformer wrappers and model declarations
- {mod}`deckard.frameworks.transformers.model` — ART-compatible transformer model configuration

These modules integrate with the canonical attack, defense, and scoring
pipelines for adversarial robustness studies on transformer models.

### Key Features

- **Tokenizer-aware wrappers**: model declarations normalize encoded payloads for sequence tasks.
- **ART integration**: transformer models wrap into
  {class}`~art.estimators.classification.HuggingFaceClassifierPyTorch`-compatible estimators.
- **Files-only persistence**: runtime output paths stay in `files={...}` aliases like other frameworks.
- **Shared scorer contract**: classification and attack metrics flow through
  {class}`deckard.score.DefaultClassifierScorerDictConfig` and related scorer dict configs.

### Data Loading

Transformers workflows typically pair with custom torch datasets that expose
tokenized inputs and labels.

- Primary runtime data adapters remain in {mod}`deckard.frameworks.pytorch.data`
- Transformer task datasets are commonly declared through
  `examples/transformers/hf_task_datasets.py`
- Split and sampler behavior remains under the canonical data contract in
  {doc}`../data/index`

### Model Configuration

The {class}`~deckard.frameworks.transformers.model.HuggingFacePytorchModelConfig`
extends torch model behavior for transformer-specific ART integration.

Key behavior includes:

- wrapping transformer modules with
  {class}`~deckard.frameworks.transformers.declarations.HuggingFaceArtModelWrapper`
- preserving integer token semantics when ART pipelines invoke model wrappers
- harmonizing device behavior across CPU, CUDA, and MPS paths

See also {doc}`../attack/index`, {doc}`../model/index`, and
{doc}`../score/index` for cross-component composition.

### Attack Configuration With Transformers

Transformers workflows use the same flattened attack runtime model as the core
API: instantiate {class}`deckard.attack.base.AttackConfig` with a canonical
attack `name`, then let runtime resolution select the concrete handler.

Common transformer attack name patterns:

- Built-in ART attacks: `art.attacks.evasion.*`
- TextAttack plugin attacks: `textattack.attack_recipes.*` (see {doc}`../plugins/textattack`)
- OpenAttack plugin attacks: `OpenAttack.attackers.*` (see {doc}`../plugins/openattack`)

Example:

```python
from deckard.attack import AttackConfig

attack_cfg = AttackConfig(
  name="textattack.attack_recipes.textfooler_jin_2019.TextFoolerJin2019",
  attack_params={"split": "test"},
)

scores = attack_cfg.run(
  data=data_cfg,
  model=model_cfg,
  files=files_cfg.as_dict(),
)
```

This dispatches through {class}`deckard.plugins.textattack.attack.TextAttackConfig`
at runtime while preserving the shared {class}`deckard.attack.base.AttackConfig`
API surface (`run`, `load`, `score`, and runtime resolver wrappers).

### Persistence Contract

- Model config persistence follows the same YAML-based config save or load flow
  as other framework configs.
- Runtime transformer model-state artifacts are persisted through the same
  file-layer helpers used by torch-backed models.
- Attack and score artifacts remain canonical, including stage-aware timing and
  score dictionaries.

### Canon Runtime Contract

Transformers configs participate in the same canonical runtime guarantees as
core and plugin families:

- files-only persistence through `files={...}` aliases
- canonical timing keys in `times`
- split-scoped score modes (`train|test|val|all`)
- stage lifecycle hook orchestration owned by the core runtime

See {doc}`../data/index` and {doc}`../data/pipeline` for canonical
stage/scope semantics.

## Examples

```{seealso}

Notebook-based transformers workflows are documented in:

- {doc}`notebooks/huggingface.ipynb </notebooks/huggingface>`

```

### Troubleshooting

- **Tokenizer or shape mismatch**: ensure dataset wrappers emit the expected
  encoded inputs for the selected model declaration.
- **Device mismatch errors**: verify all runtime components target compatible
  devices.
- **ART compatibility**: confirm optional ART and transformer dependencies are
  installed for Hugging Face model wrappers.

### See also

- {doc}`../data/index` — canonical data configuration behavior
- {doc}`../model/index` — canonical model and defense lifecycle
- {doc}`../experiment/index` — orchestration across data/model/attack/score
- {doc}`../attack/index` — attack configuration and execution
- {doc}`../../overview/extensions/transformers` — high-level transformers extension overview
