# Config Contract

Detailed contract for public config classes (`*Config`).

## Purpose

Config classes are the main control plane for Deckard runtimes. They let users
compose datasets, models, defenses, and scoring pipelines declaratively while
keeping execution deterministic.

## Capabilities

- Capture runtime knobs in serializable fields.
- Support Hydra override workflows and reproducible experiment replay.
- Encapsulate extension-specific defaults behind a stable public API.

## Standards Followed

- Documentation standards: {doc}`/developers/contributor/documentation`
- Naming rules: {doc}`/developers/contributor/naming`
- Design boundaries: {doc}`/developers/design/design`

## Canonical Contract Decisions

- Canonical identity fields use `name`; legacy constructor aliases are removed in active hard-cut slices.
- Defense runtime mappings use name and canonical runtime methods (apply, apply_to, apply_defense).
- Unknown init kwargs remain warning-first where duck-typed integration is required.
- Config finalization preserves explicit user values while deriving reproducibility-critical fields last.
- Coercion/default precedence is child normalize -> validate -> apply defaults -> compose to parent -> parent finalization.

## Config vs Mixin vs Plugin vs Subobject

Use these boundaries when introducing new runtime behavior.

### Use a `*Config` when

- Behavior needs a stable public constructor surface (`name`, params, files, scorer, mode/stage knobs).
- Behavior owns orchestration or lifecycle sequencing across stages.
- Behavior must be reproducibly serializable through YAML/Hydra and persisted runtime state.

Examples:

- [DataConfig](../../api/data/index), [ModelConfig](../../api/model/index), [AttackConfig](../../api/attack/index), [DetectorConfig](../../api/detector/index), [ExperimentConfig](../../api/experiment/index).

### Use a mixin when

- The logic is shared across multiple owners and does not need its own user-facing config schema.
- The capability is composable and narrow (normalization, state propagation, hook wiring).
- The mixin does not become the runtime owner of stage flow.

Examples:

- score/hook orchestration helpers reused by multiple config runtimes.

### Use a plugin when

- The behavior is optional, environment-dependent, or external-integration specific.
- You need explicit hook extension points without changing base config contracts.
- Feature gating or optional dependency boundaries must stay explicit.

Examples:

- text and fairness integrations that attach through declared hook surfaces.

### Use a subobject when

- The object is a bounded part of a parent config runtime (trainer, scorer, defense pipeline step, sampler).
- The object should not own global orchestration ordering.
- The parent config still controls stage/mode routing and persistence contracts.

### Quick decision rule

If it owns lifecycle and public runtime identity, make it a `*Config`.
If it shares capability across owners, make it a mixin.
If it extends behavior at optional boundaries, make it a plugin.
If it is a component inside a config-owned flow, make it a subobject.

## Required Sections

Use MyST-native Google-style sections:

- `Attributes:` required
- `Args:` when constructor/runtime parameters exist
- `Returns:` for non-`None` methods
- `Raises:` for explicit failures
- `Note:` for persistence/runtime side effects
