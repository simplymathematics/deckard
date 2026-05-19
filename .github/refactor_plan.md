# Deckard Refactor Plan

Repository: https://github.com/simplymathematics/deckard/tree/refactor-squashed

---

# Goals

1. Make `examples/*/configs` the canonical source of all configuration declarations.
2. Remove hardcoded ConfigStore registrations from Python declaration files.
3. Dynamically generate Hydra `ConfigStore` entries at runtime using `safe_store`.
4. Standardize default config naming conventions.
5. Simplify tests to compose from canonical configs instead of redefining fixtures repeatedly.
6. Ensure all `deckard/*/` modules participate in the same declaration lifecycle.
7. Enable runtime extensibility via `DECKARD_CONFIG_DIRS`.
8. **Entirely decouple `deckard/plugins/` and `deckard/frameworks/` from the abstract core so that each plugin family and framework can be tested in isolation**.
9. **All public API docstrings must be MyST-native Google-style** so they render correctly through `sphinx.ext.napoleon` + `myst_nb` without any reStructuredText markup. This makes the source readable as plain Markdown and keeps the generated HTML/notebook docs in sync with the code.
10. **Standardize attribute registration and visibility boundaries** by explicitly distinguishing internal runtime `_attributes` (class-private orchestration state) from public attributes (stable external API). Internal `_attributes` must never be relied on outside their defining class/module, and all externally consumed state must be exposed through typed public attributes or documented accessors. **Adapters are explicitly bound by this rule**: adapters MUST only read from and write to public (non-underscore-prefixed) attributes of the target config object. Private `_attributes` MAY be used inside the adapter method body as local computation variables, but MUST NOT be read from or written to the target object. Any state the adapter needs to exchange with the target MUST flow through documented public fields or typed accessor methods.


# Instructions:

Implement this from top to bottom and use `[ ]` -> `[ x]` to track your progress. 
Update the #file:refactor_plan.md. Use strikethrough syntax instead of deletion if you change anything. 
If you need further user input add it to #sym:# User Questions  and I will update it. Move on, if possible, in the case of a blocker.
---


# Canonical Nomenclature

This section defines the canonical naming conventions, object responsibilities, execution semantics, and enforcement requirements for all configuration, plugin, and composition objects throughout the repository.

These conventions are intended to:

- standardize Hydra registration behavior
- clarify public vs internal APIs
- simplify runtime config discovery
- simplify testing and composition
- make plugin execution deterministic
- make extension behavior inspectable/documented
- enforce explicit orchestration boundaries
- reduce implicit runtime behavior

---

# Core Architectural Principles

## 1. Config Objects Are Public Orchestration APIs

All `*Config` objects are:

- top-level APIs
- public-facing interfaces
- Hydra composition roots
- deterministic orchestration boundaries
- explicit execution pipelines

Every `*Config` MUST:

- inherit from `ConfigBase`
- expose explicit typed fields
- implement a public `__call__` as the orchestration entrypoint
- orchestrate plugins in a fixed execution order
- avoid hidden runtime mutation
- compose deterministically
- be independently composable/testable
- keep `__call__` orchestration readable and explicit:
  - `_run_plugin_hook()` is an allowed private orchestration bridge inside `__call__`
  - all other orchestration steps in `__call__` should be expressed through clearly named public methods
- keep `__post_init__` initialization readable and explicit through private helper methods (`_initialize_*`, `_validate_*`, `_normalize_*`, etc.)

---

# Core Abstraction Cleanup

## Goal

Refactor the existing core layer so that `deckard/frameworks/core.py` contains
only clean abstract contracts, while the `deckard/data/`, `deckard/model/`,
`deckard/attack/`, `deckard/detector/`, and `deckard/score/` base modules act
as thin orchestration boundaries instead of mixing contracts, runtime adapters,
and optional dependency wiring.

## Refactor checklist

- [x] Audit `deckard/frameworks/core.py` and align the abstract contracts for data, model, attack, detector, experiment, and scorer configs.
- [x] Remove import-time declaration loading from core package `__init__.py` modules.
- [x] Make package-level exports side-effect free when importing the core layer.
- [x] Decouple all plugin logic from the core, remove plugin shims (anjana, fairlearn, lifelines, etc.), and update all imports to reference plugins directly.
- [x] Split shared runtime state out of `deckard/data/base.py` and `deckard/model/base.py` into smaller helper or mixin modules (data pipelines, data samplers, dataloaders) and (pretrained, normal, optuna prune).
- [x] Split scorer-specific runtime helpers out of `deckard/score/base.py` so scoring contracts stay focused on orchestration.
- [x] Reduce `Any`-typed runtime payloads in core config classes by introducing typed accessors or structured state objects.
- [-] Verify `deckard/attack/base.py`, `deckard/detector/base.py`, and `deckard/experiment/base.py` only contain behavior that belongs to the core abstraction boundary.
- [x] Add regression tests proving each core package imports without eagerly loading sibling plugin families.
- [x] Add focused tests for the `deckard/frameworks/core.py` abstract contracts so concrete framework configs are checked against the shared interface.
- [x] Document which behaviors stay in the core layer versus which move into plugin or framework implementations.
- [x] Audit all adapter methods in `deckard/frameworks/adapters.py` and replace every read/write of a `_private` attribute on the target config with an equivalent public attribute or typed accessor — no private state crossing the adapter boundary.
- [x] Add public typed accessors to core config classes (`DataConfig`, `ModelConfig`, `AttackConfig`, `DetectorConfig`, `ExperimentConfig`, `ScorerDictConfig`) for every piece of internal state currently accessed by adapters (e.g. replace `_X`/`_y` reads with a `raw_data` property, `_model` reads with a `fitted_estimator` property, etc.).
- [x] Add adapter-boundary tests asserting no `_private` attribute of the target config object is accessed or mutated by any adapter mixin method during the lifecycle.

### Iterative TODO (2026-05-15) — Core Abstraction Outcomes

- [x] Outcome 1: Add regression coverage for core package imports without eager sibling plugin-family loading.
- [x] Outcome 2: Add focused framework-core contract tests that verify concrete core runtime configs conform to shared contracts.
- [x] Outcome 3: Document core-vs-framework-vs-plugin behavior boundaries.
- [x] Outcome 4: Complete adapter private-attribute audit and remove remaining private adapter helper usage.
- [x] Outcome 5: Add/verify typed public core-state accessors used by adapter-facing APIs.
- [x] Outcome 6: Keep adapter-boundary test coverage active and passing for no-private-target-access enforcement.

### Immediate TODO (2026-05-15) — Coverage Unblockers



- [ ] Remove all legacy public accesses. All run time containers and objects should only be mutated by a single *Config.
  - Audit note (2026-05-18): `DataConfig` still exposes `X_train`/`y_train`/`X_test`/`y_test` as public dataclass fields that external code writes directly. `_X`/`_y` internal fields remain. Attack code (`attack/base.py:973`) writes `model.classes_` and reads `data.X_train` etc. directly. No centralized mutation guard in place yet.
- [x] Fix attack sensitive-wrapper initialization regressions (`test_model_with_sensitive_features_predict_wraps_with_wrapper`, `test_sensitive_features_fallback_uses_sensitive_train`) in both base and pytorch attack suites.
  - Verified (2026-05-18): both tests pass in `test_attack/test_attack.py` and `test_frameworks/test_pytorch/test_pytorch_attack.py`. `SensitiveFeaturesWrapper.__init__` and `DataConfig.sensitive_test`/`sensitive_train` property setters are in place. No regressions found.
- [x] Resolve missing `require_fairlearn` fixture errors in pytorch attack fairlearn scorer tests (either register fixture globally or gate tests consistently).
  - Status (2026-05-18): ~~`require_fairlearn` fixture is now defined globally in `test/conftest.py` (line 75). The 6 ERROR entries in `test_output.txt` predate this fix. Fix is deployed; needs a fresh test run to confirm resolution. Root cause was `unittest.TestCase`+`@pytest.mark.usefixtures` requiring a conftest-level fixture rather than a module-level one.~~ ~~Fresh run of `pytest test/test_frameworks/test_pytorch/test_pytorch_attack.py` still reports the same six `require_fairlearn` fixture errors, so the fixture is not visible to this test module yet. The next step is to move or re-scope the fixture to a conftest that covers `test/test_frameworks/test_pytorch/`.~~ Re-scoped the helper as the shared `TinyFairness` example dataset in `test/helpers.py` and re-exported it from `test/test_frameworks/test_pytorch/conftest.py`; `pytest test/test_frameworks/test_pytorch/test_pytorch_attack.py -k Fairlearn -q` now passes.
- [ ] Fix framework repository enforcement blocker by resolving syntax/parse issues in `deckard/frameworks/pytorch/sample.py`.
  - Confirmed active (2026-05-18): `SyntaxError: invalid syntax` at line 81 (`n_train = ` — truncated assignment). Additionally `sample()` method references `dataset` before assignment (line 65 `if dataset is None:` but `dataset` is not a parameter). Blocks `test_repository_enforcement_frameworks_scope_passes` and coverage parsing.
- [-] Triage non-accessor coverage failures still open (`test_optimize_main_executes_once_in_multirun`, pytorch retraining defense path tests, framework lifecycle ordering assertion).
  - Triaged (2026-05-18):
    - `test_optimize_main_executes_once_in_multirun` — `KeyError: 'name'`; test asserts `captured["cfg"]["name"] == "demo"` but `optimize_main` now strips the `name` key from the injected config dict. Test expectation is stale.
    - `test_binary_input_detector_rejects_non_neural_network_models` + 3 similar — `NameError: 'DummyDataConfig' is not defined` in `test_pytorch_defenses.py`; helper class removed or never imported.
    - `test_real_adversarial_retraining_executes_with_pytorch_model` + `test_real_defensive_distillation_executes_with_pytorch_model` + `test_real_neural_cleanse_reports_backend_incompatibility_for_pytorch_model` + `test_art_last_ordering_no_warning_for_wrapper_only_chain` — `FileNotFoundError` for YAML configs at `test/examples/pytorch/config/defense/{adversarial_retraining,defensive_distillation,neural_cleanse,fairlearn-adversarial-classifier}.yaml`. These canonical YAML files do not exist under `test/examples/`.
    - `test_framework_contracts_declare_ordered_lifecycle[_StubFrameworkModelConfig-expected_execution_steps2]` — lifecycle `__post_init__` step ordering mismatch: `init_context` and `init_scoring` appear swapped relative to test expectations.
    - `test_retraining_defense_is_reordered_last_with_warning` — `AssertionError: ModuleNotFoundError not raised`; test expects the defense pipeline to raise but it only warns now.
- [-] Re-run full coverage (`scripts/coverage.sh`) and record updated pass/fail/error counts in this plan.
  - Last run (2026-05-18, from `test_output.txt`): **13 failed, 1499 passed, 8 skipped, 1 xpassed, 6 errors** in 190s. Overall coverage: **81%** (14040 stmts, 2705 missed). Key low-coverage modules: `deckard/frameworks/adapters.py` 53%, `deckard/frameworks/pytorch/score.py` 23%, `deckard/plugins/yellowbrick/plot.py` 38%, `deckard/plugins/seaborn/plot.py` 58%, `deckard/frameworks/pytorch/sample.py` excluded (parse error).
  - Focused run (2026-05-18): `pytest test/test_frameworks/test_pytorch/test_pytorch_attack.py -k Fairlearn -q` → **6 passed, 133 deselected**.


---

# Canonical Config Naming

## Standard Form

```text
<Framework><Type>Config
```

Examples:

| Purpose | Canonical Name |
|---|---|
| generic model config | `ModelConfig` |
| sklearn model config | `SklearnModelConfig` |
| pytorch model config | `PytorchModelConfig` |
| attack config | `AttackConfig` |
| defense config | `DefenseConfig` |
| experiment config | `ExperimentConfig` |

Rule:
`Framework` is reserved for abstract base contracts in `deckard/frameworks/core.py`.
Concrete implementations must use framework-prefixed names like `Pytorch*Config` and `Sklearn*Config` (for example `PytorchModelConfig`), not `PytorchFramework*Config`.

---

# Framework Namespacing Convention

## Internal vs. User-Facing Imports

**Framework implementations** live at the internal canonical path:

```text
deckard/frameworks/<framework>/
```

Example canonical paths:

```text
deckard/frameworks/pytorch/data.py
deckard/frameworks/pytorch/model.py
deckard/frameworks/sklearn/data.py
```

**User-facing re-exports** provide ergonomic access via:

```text
deckard/<framework>/
```

This enables clean user imports:

```python
# User-facing convenience (preferred)
from deckard.pytorch import PytorchModelConfig
from deckard.pytorch.data import PytorchDataConfig
from deckard.pytorch.experiment import TorchExperimentConfig

# Also available (internal canonical path)
from deckard.frameworks.pytorch.model import PytorchModelConfig
from deckard.frameworks.pytorch.data import PytorchDataConfig
```

### Re-export Strategy

1. Each `deckard/<framework>/` module (e.g., `data.py`, `model.py`) imports and re-exports all public names from `deckard/frameworks/<framework>/`.
2. This enables clean user-facing imports via `deckard.pytorch.*` while maintaining the internal canonical path at `deckard.frameworks.pytorch.*`.

---

# Config Responsibilities

## `*Config` objects MUST:

- define orchestration order
- define plugin execution order
- define runtime composition boundaries
- expose stable public APIs
- expose explicit typed dataclass fields
- support Hydra composition
- remain serializable
- avoid embedding implementation-specific runtime logic

---

# Scorer Naming Convention

## Exception: Scorers

Scorers are intentionally extensible/context-aware.

To make this explicit, scorer defaults MUST use the `Default*` prefix.

---

# Canonical Scorer Form

```text
Default<Extension>ScoreConfig
```

Examples:

| Purpose | Canonical Name |
|---|---|
| generic scorer | `DefaultScoreConfig` |
| sklearn scorer | `DefaultSklearnScoreConfig` |
| pytorch scorer | `DefaultPytorchScoreConfig` |
| fairlearn scorer | `DefaultFairlearnScoreConfig` |
| attack-aware scorer | `DefaultAttackScoreConfig` |
| model-aware scorer | `DefaultModelScoreConfig` |

---

# Scorer Rules

## `Default*ScoreConfig` objects MUST:

- be extendable
- support context-aware composition
- support override injection
- expose deterministic scoring pipelines
- remain Hydra-composable

---

# Mixin Naming Convention

## Purpose

`*_Mixin` objects are implementation-layer dataclasses that encapsulate reusable behavior, logic, and parameters.

Mixins are NOT top-level orchestration APIs.

---

# Canonical Mixin Form

```text
<Extension><Capability>Mixin
```

Examples:

| Purpose | Canonical Name |
|---|---|
| defense behavior | `DefenseMixin` |
| sklearn defense behavior | `SklearnDefenseMixin` |
| pipeline behavior | `PipelineMixin` |
| pytorch pipeline behavior | `PytorchPipelineMixin` |

---

# Mixin Responsibilities

## `*_Mixin` objects MUST:

- be dataclasses
- encapsulate reusable behavior
- encapsulate reusable parameters
- encapsulate reusable execution logic
- expose at least one public-facing method
- ~~implement top-level orchestration via `__call__`~~
- NOT implement `__call__`; mixins expose public methods that are invoked by `*Config` orchestration
- expose explicit type annotations
- contain MyST-native Google-style docstrings for all public APIs
- avoid hidden side effects

---

# Adapter Attribute Contract

## Rule

`*ContractMixin` objects (in `deckard/frameworks/adapters.py`) bridge
existing core runtime objects to abstract framework contracts. They are the
**only** permitted cross-boundary translation layer.

## Adapter MUST:

- read and write only **public** (non-underscore-prefixed) attributes of the
  target config object
- introduce public typed accessor methods on core configs for any state
  currently hidden behind private attributes
- use `_private` names **only** as local computation variables inside the
  adapter method body — MUST NOT reference `self._anything` on the target config
- expose every exchanged value through a typed public field or property so
  that the contract surface is inspectable without running the adapter

## Adapter MUST NOT:

- read `target._private_attr` from the config object
- write `target._private_attr = value` on the config object
- bypass the typed public API to fetch or mutate orchestration state
- shadow or duplicate private attributes from the target class

## Enforcement

A linting rule or test assertion MUST verify that no `ContractMixin`
method body contains an attribute access of the form `self._[a-z]` on the
target config object, unless that name is declared as a public `field(...)` or
`@property` on the config class.

---

# Public Method Requirements

## Lifecycle Readability Rule (`__call__` and `__post_init__`)

For all `*Config` classes:

- `__call__` should read as a clear public orchestration sequence.
- `_run_plugin_hook()` is explicitly allowed in `__call__` as the private plugin-bridge helper.
- Aside from `_run_plugin_hook()`, `__call__` should delegate to public, clearly named step methods.
- `__post_init__` should delegate to private setup/validation helpers only, so initialization intent remains inspectable.

## Every `*_Mixin` MUST provide:

- at least one public-facing method
- explicit type annotations
- a **MyST-native Google-style** docstring describing:
  - `Args:` — all parameters with types and purpose
  - `Returns:` — return type and semantics
  - `Raises:` — exceptions and conditions
  - `Note:` — side effects and execution assumptions

Docstring rules:
- Use Google-style sections (`Args:`, `Returns:`, `Raises:`, `Note:`, `Example:`)
- No reStructuredText markup (no `:param:`, `:type:`, `:rtype:`, `.. code-block::`)
- Inline code uses single backticks: `` `my_field` ``
- Cross-references use MyST link syntax: `` {class}`deckard.data.base.DataConfig` ``
- Rendered by `sphinx.ext.napoleon` → MyST via `myst_nb`

---

# Fallback Rule

## If no meaningful public methods exist:

The top-level execution method on `*Config` MUST:

- be public
- include a full MyST-native Google-style docstring
- include explicit type annotations
- avoid:
  - `Any`
  - `object`
  - implicit runtime contracts

---

# Forbidden Mixin Patterns

## The following are NOT allowed:

- undocumented mixins
- private-only APIs
- implicit execution contracts
- untyped runtime payloads
- hidden orchestration logic
- side-effect-only mixins

---

# Plugin Naming Convention

## Purpose

`*_Plugin` objects compose one or more mixins into deterministic executable runtime units.

Plugins are runtime execution adapters between:

- orchestration configs
- mixin behavior
- runtime execution hooks

---

# Canonical Plugin Form

```text
<Extension><Capability>Plugin
```

Examples:

| Purpose | Canonical Name |
|---|---|
| sklearn defense plugin | `SklearnDefensePlugin` |
| pytorch attack plugin | `PytorchAttackPlugin` |
| scoring plugin | `ScorePlugin` |
| data pipeline plugin | `DataPipelinePlugin` |

---

# Plugin Responsibilities

## `*_Plugin` objects MUST:

- compose one or more mixins
- define deterministic execution order
- implement runtime execution hooks
- expose a public `__call__`
- pass through `*args`
- pass through `**kwargs`
- orchestrate mixin execution deterministically
- avoid hidden runtime mutation

---

# Plugin Execution Rules

## `*_Plugin.__call__`

Plugins MUST implement:

```python
def __call__(self, *args, **kwargs):
```

and MUST:

- receive arbitrary runtime arguments
- pass arguments through execution layers
- invoke mixin logic using explicit hooks
- preserve deterministic execution ordering
- expose documented execution semantics

---

# Plugin Composition Rules

## Plugins MAY:

- compose multiple mixins
- define execution stages
- define pre/post hooks
- expose runtime adapters

## Plugins MUST NOT:

- mutate orchestration configs
- register Hydra configs directly
- bypass typed mixin interfaces
- perform implicit dependency injection

---

# Docstring Standard

## Format: MyST-Native Google Style

All public API docstrings in `deckard/` MUST use **Google-style** sections rendered
via `sphinx.ext.napoleon` → `myst_nb`.  No reStructuredText markup is allowed in
public docstrings.

### Section reference

| Section | When to include |
|---|---|
| `Args:` | any parameter |
| `Returns:` | non-`None` return value |
| `Raises:` | documented exceptions |
| `Note:` | side effects, execution assumptions |
| `Example:` | canonical usage (Markdown fenced block) |

### Syntax rules

- **No RST markup** — forbid `:param name:`, `:type name:`, `:rtype:`, `.. code-block::`, `.. note::`, etc.
- **Inline code** uses single backticks: `` `my_field` ``
- **Cross-references** use MyST role syntax: `` {class}`deckard.data.base.DataConfig` ``
- **Code examples** use fenced Markdown blocks (` ```python `), not RST directives
- `napoleon_google_docstring = True` (already set in `docs/conf.py`)
- Target: `napoleon_numpy_docstring = False` once all docstrings are migrated

### Canonical example

```python
def _sensitive_labels_from_frame(
    self,
    frame: pd.DataFrame,
) -> pd.Series:
    """Resolve sensitive labels from *frame* using `sensitive_columns`.

    Args:
        frame: Source DataFrame containing the sensitive-feature columns
            listed in `sensitive_columns`.

    Returns:
        A string-typed Series of sensitive labels aligned with *frame*.

    Raises:
        ValueError: When `sensitive_columns` is ``None`` and no fallback
            can be inferred from the target splits.
        KeyError: When any column in `sensitive_columns` is absent from
            *frame*.
    """
```

---

# Declaration Naming

## Runtime declarations

Runtime-generated declarations MUST follow:

```text
<framework>/<group>/<name>
```
or


```text
<framework>/<group>/<plugin>-<name>
```

Examples:

```text
sklearn/model/random_forest
pytorch/model/resnet18
sklearn/attack/fgsm
pytorch/defense/anjana-adult
```

---

# YAML Naming Convention

## File names

YAML filenames MUST use a modified snake case where aliases can contain '-', but categories like group/framework/plugin/etc are separated by "_":

Examples:

```text
sklearn/config/model/random-forest.yaml
pytorch/config/model/resnet18.yaml
config/score/default.yaml
config/defense/anjana_k-anonymity.yaml
config/score/fairlearn.yaml
```

## Plugin YAML Paths

Plugin YAML configs MUST use:

```text
group/<plugin>_<aliased-name>.yaml
```

Examples:

```text
data/anjana_custom.yaml
model/fairlearn-classifier.yaml
experiment/lifelines-survival.yaml
plot/yellowbrick-feature-importance.yaml
score/fairlearn-demographic-parity.yaml
```

---

# Python Class Naming Convention

## Python objects

Python config classes MUST use:

```python
PascalCase
```

Examples:

```python
RandomForestModelConfig
Resnet18ModelConfig
DefaultScoreConfig
AnjanaDefensePlugin
PipelineMixin
```

---

# Hydra Group Naming

## Canonical group structure

```text
<domain>/<type>/<implementation>
```

Examples:

```text
model/sklearn/random-forest
model/pytorch/resnet18
attack/evasion/fgsm
defense/preprocessor/anjana
pipeline/train/default
```

---

# Runtime Discovery Naming

## Discovery roots

Canonical runtime roots:

```text
examples/sklearn/configs
examples/pytorch/configs
```

External roots:

```text
DECKARD_CONFIG_DIRS
```

---

# Public vs Internal APIs

## Public APIs

Public APIs MUST use:

```text
*Config
*Plugin
```

Examples:

- `ModelConfig`
- `AttackConfig`
- `ScorePlugin`

These are stable runtime interfaces.

---

## Internal Composition APIs

Internal reusable behavior MUST use:

```text
*Mixin
```

Examples:

- `DefenseMixin`
- `PipelineMixin`

These are implementation-layer abstractions.


# Canonical Object Matrix

| Object Type | Public | Registered | Runtime Generated | User Facing |
|---|---|---|---|---|
| `*Config` | Yes | Yes | Yes | Yes |
| `Default*ScoreConfig` | Yes | Yes | Yes | Yes |
| `*Plugin` | Yes | No | Yes | Yes |
| `*Mixin` | No | No | No | No |

---

# Future Extension Rules

New frameworks/plugins MUST:

- expose top-level `*Config` objects
- expose deterministic `*Plugin` execution
- avoid exposing mixins publicly
- register through runtime discovery
- provide YAML declarations in canonical config roots
- avoid hardcoded ConfigStore registration
- expose typed/documented public APIs

---

# Non-Goals

The following MUST NOT exist after refactor completion:

- ad-hoc declaration naming
- undocumented mixins
- hidden plugin execution order
- implicit runtime orchestration
- untyped runtime APIs
- declaration-only Python files
- duplicate config APIs
- hardcoded ConfigStore registration
- implicit plugin side effects
- hidden dependency injection
- ambiguous scorer defaults
- backwards-compatibility shim modules used instead of fixing broken imports

# Architectural Direction

## Canonical Config Source

Canonical config definitions must live in:

- `examples/sklearn/configs/**`
- `examples/pytorch/configs/**`

Optional external config roots:

- `$DECKARD_CONFIG_DIRS`

The `deckard/` package should no longer contain authoritative config declarations.

---

# Runtime Declaration Architecture

## New File

Create:

```text
deckard/declarations.py
```

Responsibilities:

- Discover config directories
- Parse YAML configs
- Register Hydra ConfigStore entries dynamically
- Use `safe_store`
- Register only compatible/installed integrations
- Support external plugin config roots

---

# Naming Convention

All defaults must follow:

```text
Default<Extension><Type>Config
```

Examples:

| Config | Name |
|---|---|
| default defense | `DefaultDefenseConfig` |
| default anjana defense | `DefaultAnjanaDefenseConfig` |
| default sklearn model | `DefaultSklearnModelConfig` |
| default pytorch attack | `DefaultPytorchAttackConfig` |
| default scorer | `DefaultScoreConfig` |

Scorers must become context-aware:

- data-aware
- model-aware
- attack-aware
- classification-aware

---

# Runtime Config Discovery

## Discovery Order

### Built-in canonical roots

```python
examples/sklearn/configs
examples/pytorch/configs
```

### External roots

```bash
DECKARD_CONFIG_DIRS=/path/a:/path/b
```

---

# Proposed `deckard/declarations.py`

## Responsibilities

### 1. Discover config roots

```python
def discover_config_roots() -> list[Path]:
```

### 2. Enumerate YAML declarations

```python
def iter_config_files(root: Path):
```

### 3. Parse declarations safely

```python
safe_store(...)
```

### 4. Register with Hydra ConfigStore

```python
ConfigStore.instance().store(...)
```

### 5. Conditionally register integrations

Example:

```python
if importlib.util.find_spec("torch"):
    register_pytorch_configs()
```

---

# Runtime Registration Flow

## Package startup

During:

```python
deckard.__main__
```

or package initialization:

```python
deckard/declarations.py
```

the system should:

1. discover config roots
2. parse YAML configs
3. register ConfigStore objects
4. expose defaults automatically

---

# Testing Strategy Refactor

## Current Problem

Tests repeatedly redefine:

- data configs
- model configs
- attack configs
- scorer configs

This duplicates canonical declarations.

---

# New Testing Hierarchy

## Level 1 — Hydra Compose Tests

Validate:

- configs compose correctly
- overrides resolve correctly
- defaults resolve correctly
- Compose test command shape: `deckard optimize <hydra overrides> --cfg job`
- Use a curated representative matrix only (do not sweep all combinations)

Fastest layer.

---

## Level 2 — Unit Tests

Validate:

- individual modules
- parsers
- factories
- adapters
- scorers

Use canonical configs via compose.

---

## Level 3 — Experiment Tests

Validate:

- end-to-end pipelines
- sklearn workflows
- pytorch workflows
- Experiment test command shape: execute `deckard optimize <hydra overrides>` (run or multirun)
- Use selected smoke combinations that exercise core paths; avoid Cartesian-product sweeps

Only after compose/unit tests pass.

### Representative Validation Matrix (Non-Exhaustive)

Run a small, stable set of combinations that covers code paths without combinatorial explosion:

- sklearn baseline classification (`data=test-classification`, `model=test-logistic`, `attack=hsj`, `defense=class-labels`, `score=classification`)
- sklearn regression path (`data=regression`, `model=ridge`, `attack=none`, `score=regression`)
- pytorch baseline classification (`data=torch_mnist`, `model=default`, `attack=fgm`, `score=pytorch_classification`)
- one plugin-aware path per plugin family in scope (for example fairlearn, anjana, lifelines)

Do not attempt full cross-product validation across all config groups.

---

# Refactor TODO Matrix

The following TODO process must be repeated for every `deckard/*/` subfolder before integration tests are enabled.

---

# Agent Customization Draft

This plan implies a specialized repository agent for deterministic config-refactor work.

## Extracted Agent Role

- specialized role/persona: repository refactor planner and executor for Hydra/config declaration migration
- preferred tools: targeted single-file edits, minimal read-before-edit, avoid unrelated files and broad repository changes
- domain/job scope: canonical config migration, runtime declaration registration, naming enforcement, and test simplification

## Proposed `.agent.md`

```markdown
---
name: deckard-refactor
description: Use for updating the Deckard refactor plan and making tightly scoped config-refactor edits that preserve canonical naming, runtime declaration loading, and checklist progress.
~~model: GPT-5.4~~
model: GPT-5.3-Codex
tools:
  - read_file
  - apply_patch
---

# Purpose

You are a focused Deckard refactor agent.

Use this agent when work involves:

- Hydra config declaration migration
- `examples/*/configs` canonicalization
- `safe_store` runtime registration planning
- naming convention enforcement
- refactor checklist maintenance

# Operating Rules

- Prefer small, direct edits.
- Only change files explicitly requested by the user.
- If editing a plan/checklist, update progress using `[ ]` -> `[ x]`.
- Use strikethrough instead of deleting superseded plan text.
- Avoid speculative repository-wide changes.
- Keep naming aligned with:
  - `*Config`
  - `Default*ScoreConfig`
  - `*Mixin`
  - `*Plugin`
  - *Pipeline* Data modifiers
  - *Defense* Model modifiers
- Preserve deterministic, runtime-driven declaration architecture.

# Tool Preferences

- Use `read_file` only for the target file when more context is required.
- Make all requested text/code edits in a single `apply_patch` call.
- Do not modify unrelated files.

# Output Style

- Be brief and implementation-focused.
- When blocked, add concise questions under `# User Questions` if the target file is a plan.
- Prefer actionable edits over explanation.
```

## Ambiguities To Confirm

- Should the agent be limited to planning/checklist updates, or also used for code refactors in `deckard/` and `examples/`?

The agent should implement the refactors and testing as described.

- Should the agent explicitly avoid multi-file changes, or allow them when the user requests a coordinated refactor?

Follow the instructions in order. If these are unclear, minimize scope and then expand scope until the instructions are satisfied.

- Should Hydra-specific validation/testing commands be embedded in the agent instructions?


Yes. tests should include Config composition, config calling, and config declarations using the existing .yaml files and the appropriate framework context DECKARD_CONFIG_DIR and DECKARD_DEFAULT_CONFIG_FILE. In addition, be sure to include sub command tests using:

`deckard optimize <hydra overrides>` with a blank default.yaml file context (so that sklearn/ torch defaults don't destabilize tests).

Minimize test overlap and test run-time as secondary goal.


## Suggested Follow-Up

- [x] Draft agent specialization from this plan
- [x] Confirm agent scope boundaries
- [x] Confirm tool allow/avoid preferences
- [x] Finalize `.agent.md` for repository use
- [x] Create specialized agents in `.github/agents/`

## Agent Finalization Summary


- **deckard-refactor.agent.md**: Primary agent for config-refactor work
  - Role: focused Deckard refactor planner/executor for Hydra config declaration migration and test simplification
  - Tools: read, edit, search, todo (minimal, targeted)
  - Use when: canonical config migration, runtime declaration registration, naming enforcement, checklist progress
  - Scope: tightly scoped to refactor-specific edits, preserves canonical naming, supports both planning and implementation

- **deckard-runtime-declarations.agent.md**: Runtime discovery/parsing/registration specialist
  - Implements config discovery, YAML parsing, safe_store registration, optional dependency handling
  - Coordinates with deckard-refactor for Phase 0 infrastructure setup

- **deckard-naming-enforcer.agent.md**: Naming convention validation specialist
  - Enforces *Config, Default*ScoreConfig, *Mixin, *Plugin conventions
  - Adds linting/CI checks for compliance
  - Used after modules are consolidated

- **deckard-test-minimizer.agent.md**: Test optimization specialist
  - Consolidates fixture configs into canonical YAML
  - Minimizes test layer overlap (compose, unit, experiment)
  - Used after config consolidation to clean up tests

## Example Prompts to Use With `deckard-refactor`

- "Implement Phase 0: create `deckard/declarations.py` and wire runtime config discovery from `examples/*/configs`"
- "Refactor `deckard/model/` to use runtime `safe_store` registration—consolidate configs to YAML and update checklist"
- "Create compose-first tests for `deckard/data/` canonical YAML configs"
- "Update refactor plan progress for Phase 1 completion"

When work is more specialized, delegate:
- Use `deckard-runtime-declarations` for Phase 0 `deckard/declarations.py` implementation details
- Use `deckard-naming-enforcer` for validation and CI checks after modules are refactored
- Use `deckard-test-minimizer` for test consolidation after configs are in place

## Related Customizations Created

- [x] deckard-refactor agent (main)
- [x] deckard-runtime-declarations agent (Phase 0 support)
- [x] deckard-naming-enforcer agent (enforcement phase support)
- [x] deckard-test-minimizer agent (test cleanup support)

---

# Repository-Wide TODO

## Phase 0 — Core Infrastructure

- [x] Create `deckard/declarations.py`
- [x] Add runtime config discovery
- [x] Add `DECKARD_CONFIG_DIRS` support
- [x] Add YAML parsing helpers
- [x] Add `safe_store` integration
- [x] Add installed-package detection
- [x] Add Hydra ConfigStore runtime registration
- [x] Add logging/debug visibility for discovered declarations
- [x] Add duplicate registration detection
- [x] Add validation for malformed YAML declarations
- [x] Add tests for runtime declaration loading

---

# Phase 1 — Canonical Config Consolidation

## Global Tasks

- [x] Move all authoritative configs into:
  - [x] `examples/sklearn/configs`
  - [x] `examples/pytorch/configs`
- [x] Ensure every existing declaration has a YAML equivalent
- [x] Remove duplicated declaration logic from Python files
- [x] Normalize naming conventions
- [x] Normalize defaults structure
- [x] Normalize config groups
- [x] Ensure every config composes independently

---

# Per-Module Refactor TODO

Repeat this checklist for each `deckard/*/` subfolder.

Module test gate policy:
- In module blocks (`deckard/data`, `deckard/model`, `deckard/attack`, `deckard/score`, `deckard/plot`, `deckard/experiment`, etc.), only **base objects** are required to pass.
- Plugin-family objects are explicitly deferred to and validated in the `deckard/plugins/` step.

---

# Concurrent with modules:

- [x] Add family-specific plugin wrapper modules under `deckard/plugins/`
- [x] Add import smoke tests for plugin package facades and family modules
- [x] Add package wrapper modules for `deckard/data/pipeline/` and `deckard/model/defense/`
- [x] Add import smoke tests for data-pipeline and model-defense wrappers
- [x] Add framework namespace aliases under `deckard/frameworks/`
- [x] Add import smoke tests for framework namespace aliases
- [x] Add canonical family aliases for `deckard/score/`
- [x] Add import smoke tests for score family aliases
- [x] Add canonical family aliases for `deckard/plot/`
- [x] Add import smoke tests for plot family aliases
- [x] Add canonical family aliases for `deckard/experiment/`
- [x] Add import smoke tests for experiment family aliases
- [x] Add canonical family aliases for `deckard/data/` and `deckard/model/`
- [x] Add import smoke tests for data and model family aliases
- [x] Continue package-family wrapper migration for remaining module groups
- [x] Split data-pipeline wrappers through a core module to remove circular imports
- [x] Clean up duplicated model export imports introduced by wrapper migration
- [x] Identify shared logic across plugin families (audit)
- [x] Extract framework-independent `_SensitiveColumnsMixin` into `deckard/data/_mixins.py` (fields: `sensitive_columns`, `fairness_defense`; methods: `_sensitive_labels_from_*`, `_validate_sensitive_runtime`)
- [x] Update `deckard/data/fairness._SensitiveColumnsMixin` to extend `_SensitiveColumnsMixin` (fairlearn-specific methods remain: `_inject_fairness_defense_step`, `_sample`)
- [x] Update `deckard/data/anjana.AnjanaDataConfig` to inherit `_SensitiveColumnsMixin` directly (breaks anjana→fairlearn data-layer coupling)
- [x] Move `test_inject_fairness_defense_step_branch_paths` from `test_anjana_data_unit` to `test_fairness_data_unit` (was testing fairlearn-specific behaviour via anjana)

# Enforcement TODO

## Repository-Wide Enforcement

- [x] Enforce `*Config` naming convention
- [ ] Enforce `Default*ScoreConfig` naming convention
- [x] Enforce `*Mixin` naming convention
- [x] Enforce `*Plugin` naming convention
- [x] Ensure all `*Config` inherit from `ConfigBase`
- [x] Ensure all `*Mixin` objects are dataclasses
- [x] Ensure all `*Plugin` objects implement `__call__`
- [x] Ensure all plugins pass through `*args` and `**kwargs`
- [x] Ensure all plugins define deterministic execution ordering
- [x] Ensure all mixins expose at least one public-facing method
- [ ] Ensure all public methods include MyST-native Google-style docstrings (no rST markup)
- [ ] Ensure all public methods use explicit type annotations
- [ ] Remove usages of
  - [ ] `Any`
  - [ ] `object`
  - [ ] implicit runtime payloads
   AND PREFER deckard OBJECTS
- [ ] Enforce canonical YAML naming
- [ ] Enforce canonical Hydra group naming
- [x] Add static validation for nomenclature violations
- [x] Add CI checks for naming convention enforcement
- [x] Add CI checks for missing docstrings (pydocstyle or ruff D-rules)
- [x] Add CI checks for non-Google-style docstring sections (forbid `:param:`, `:type:`, `.. code-block::`)
- [x] Configure `napoleon_google_docstring = True`, `napoleon_numpy_docstring = False` in `docs/conf.py` once all docstrings are Google-style
- [x] Add CI checks for missing type annotations
- [x] Add runtime validation for plugin execution ordering
- [x] Add tests validating deterministic plugin orchestration
- [x] Add tests validating mixin composition order
- [x] Add tests validating runtime declaration consistency
- [x] Enforce adapter public-attribute contract: no adapter method may read or write a `_private` attribute on the target config object
- [x] Add linting rule or test guard asserting adapter boundary cleanliness (no `self._` accesses on the target that are not declared public fields/properties)
- [x] Introduce typed public accessor properties on core config classes to replace all private-attribute reads currently performed by adapter mixins

### Enforcement Progress Snapshot (2026-05-14)

- [x] Blocking enforcement remains green for both scopes:
  - `python scripts/repository_enforcement.py --scope deckard/plugins`
  - `python scripts/repository_enforcement.py --scope deckard/frameworks`
- [x] Strict-docs-types for `deckard/frameworks` reduced from 112 to 0 violations.
- [x] Strict-docs-types for `deckard/plugins` reduced from 107 to 82 violations.
- [x] Framework scorer/context contracts now use typed context parameters (`DataConfig`, `ModelConfig`, `AttackConfig`) instead of generic runtime placeholders where applicable.
- [x] Model-defense adapter contract now types `apply_to(estimator, data)` as `ModelConfig` + `DataConfig` context (no generic `RuntimeValue` estimator slot).
- [x] Variadic runtime payload parameters (`*args`, `**kwargs`) are explicitly permitted to remain `Any` by enforcement policy.
- [x] Framework adapter signatures now use explicit core-aligned named types (`DataConfig`, `ModelConfig`, `AttackConfig`, `ArrayLike`, `RuntimeValue`) with permissive variadics only.
- [x] Data-contract `X`/`y` signatures are now explicit `MatrixLike`/`ArrayLike` across framework contracts and adapters, with install-aware runtime type registration hooks for optional framework classes.
- [x] Model-defense `estimator` is now `EstimatorLike` (framework runtime object) with install-aware defaults including sklearn estimators, torch modules, and ART sklearn/pytorch wrappers.
- [x] Core scoring base now adapts to `FrameworkDataScorer` with explicit `score(ind: MatrixLike, dep: ArrayLike, ...)` bridge semantics.
- [x] Core `data/model/attack` base modules now expose tighter framework-facing public signatures (`MatrixLike`/`ArrayLike`/`EstimatorLike`) on key bridge methods while preserving runtime behavior.
- [x] Completed strict hardening for:
  - `deckard/frameworks/core.py`
  - `deckard/frameworks/adapters.py`
  - `deckard/frameworks/pytorch/{data,model,experiment,fairness_data}.py`
  - `deckard/plugins/{anjana,fairlearn}/**`
- [ ] Remaining strict hotspots are concentrated in:
  - `deckard/plugins/lifelines/{experiment,model,plot}.py`
  - `deckard/plugins/{seaborn,yellowbrick}/plot.py`

## Shared Mixin Infrastructure

- [x] Audit all plugin families for shared logic
- [x] Create `deckard/data/_mixins.py` with `_SensitiveColumnsMixin` — framework-independent sensitive-column fields and helpers shared by both anjana and fairlearn data configs
- [x] `data/fairness._SensitiveColumnsMixin` now extends `_SensitiveColumnsMixin`; fairlearn-specific methods (`_inject_fairness_defense_step`, `_sample`) stay in fairlearn
- [x] `data/anjana.AnjanaDataConfig` now inherits `_SensitiveColumnsMixin` directly — breaks anjana→fairlearn data-layer coupling
- [x] Audit `deckard/model/` for shared cross-plugin logic and extract if overlap is found
- [x] Integrate shared model runtime mixins into `ModelConfig` (`ModelTrainingMixin`, `PretrainedModelMixin`, `ModelPrunerMixin`) and route `_train` through mixin training behavior
- [x] Add focused mixin integration tests for `data` and `model` runtime entrypoints (`test/test_data/test_data_mixins.py`, `test/test_model/test_model_mixins.py`)
- [x] Audit `deckard/score/` for shared scoring helpers across plugin families
- [x] Add isolation import tests: each plugin family importable without sibling families
  - [x] Added score-family isolation imports in `test/test_package/test_plugin_family_isolation.py` for anjana, fairlearn, and lifelines score plugins.


# TODO — `deckard/data/`

- [x] Audit all dataset loading functions including optional dependencies
- [x] Audit all ConfigStore declarations
- [x] Ensure consistent nomenclature
- [x] Ensure equivalent YAML exists in canonical configs
- [x] Consolidate sklearn-compatible configs
- [x] Consolidate pytorch-compatible configs
- [x] Add parsing support to `deckard/declarations.py`
- [x] Register via runtime `safe_store`
- [x] Remove `*declaration*.py` hardcoded registrations
- [x] Remove hardcoded ConfigStore usage
- [x] Add compose tests
- [x] Update unit tests to use canonical configs
- [x] Update experiment tests
- [x] Ensure base data tests pass (plugin objects deferred to `deckard/plugins/`)

---

# TODO — `deckard/model/`

- [x] Audit all ConfigStore declarations
- [x] Ensure equivalent YAML exists in canonical configs
- [x] Consolidate sklearn configs
- [x] Consolidate pytorch configs
- [x] Normalize model defaults
- [x] Add runtime parsing support
- [x] Register via `safe_store`
- [x] Remove declaration Python files hardcoded registrations
- [x] Add compose tests
- [x] Refactor unit tests
- [x] Refactor experiment tests
- [x] Ensure base model tests pass (plugin objects deferred to `deckard/plugins/`)

---

# TODO — `deckard/attack/`

- [x] Audit attack  ConfigStore declarations 
- [x] Ensure YAML equivalents exist
- [x] Consolidate canonical attack configs
- [x] Normalize attack defaults
- [x] Add runtime parsing support
- [x] Register dynamically
- [x] Remove declaration Python files
- [x] Add compose tests
- [x] Refactor attack unit tests
- [x] Refactor experiment tests
- [x] Ensure base attack tests pass (plugin objects deferred to `deckard/plugins/`)
- [ ] Ensure consistent use of scoring mode. 
- [ ] Ensure that scoring mode is context aware (attack-type depdendent)


---

# TODO — `deckard/model/defense/`
- [x] Create new folder
- [x] Audit defense declarations (Art defence types, fairlearn model defences (reductions/postprocessing))
- [x] Ensure YAML equivalents exist
- [x] Normalize defense defaults
- [x] Add `DefaultDefenseConfig`
- [x] Add framework-specific defaults
- [x] Add runtime parsing support
- [x] Register dynamically
- [x] Remove declaration Python files
- [x] Add compose tests
- [x] Refactor unit tests
- [x] Refactor experiment tests
- [x] Ensure base defense tests pass (plugin objects deferred to `deckard/plugins/`)
- [ ] Enforce the _Mixin -> _Plugin -> Config rule using the new abstract base class and implement overrided methods as needed

---

# TODO — `deckard/score/`

- [x] Audit scorer declarations
- [x] Ensure YAML equivalents exist
- [x] Add context-aware scorer configs
- [x] Implement:
  - [x] data-aware scorers
  - [x] model-aware scorers
  - [x] attack-aware scorers
  - [x] classifier-aware scorers
- [x] Add `DefaultScoreConfig`
- [x] Add runtime parsing support
- [x] Register dynamically
- [x] Remove declaration Python files
- [x] Add compose tests
- [x] Refactor unit tests
- [x] Refactor experiment tests
- [x] Ensure base scoring tests pass (plugin objects deferred to 
- [x] Change y_pred, y_true nomenclature to dep, ind everywhere to avoid confusion with Data-only scoring
- [ ] Allow a user to configure data, model, or attack-only scorers
- [ ] Allow a scoring mode for pre/post-defense
- [ ] Allow a post-sample scoring mode (.X, .y after transform)

`deckard/plugins/`)

---

# TODO — `deckard/plot/`

- [x] Audit plot declarations
- [x] Ensure YAML equivalents exist
- [x] Add runtime parsing support
- [x] Register dynamically
- [x] Remove declaration Python files
- [x] Add compose tests
- [x] Refactor unit tests
- [x] Ensure base plot tests pass (plugin objects deferred to `deckard/plugins/`)

---



---

# TODO — `deckard/data/pipeline/`

- [x] Audit pipeline declarations (DataPipeline AnjanaDataPipeline FairlearnDataPipeline)
- [x] Ensure YAML equivalents exist
- [x] Consolidate canonical pipeline configs
- [x] Normalize defaults
- [x] Add runtime parsing support
- [x] Register dynamically
- [x] Remove declaration Python files
- [x] Add compose tests
- [x] Refactor pipeline tests
- [x] Refactor experiment tests
- [ ] Enforce the _Mixin -> _Plugin -> Config rule using the new abstract base class and implement overrided methods as needed

---

# TODO — `deckard/frameworks/`

**Goal**: Decouple each framework from the abstract core and from each other so that each `deckard/frameworks/<framework>` can be instantiated, composed, and tested in isolation without importing any sibling framework. 

- [x] Create `deckard/frameworks/sklearn` and `deckard/frameworks/pytorch`.. 
- [x] Create abstract base <data/model/attack/detector/experiment/scorer/>Configs that declare necessary functions and attributes in the top level *base.py files. Enforce the _Mixin -> _Plugin -> Config rule using this new abstract base class and implement overrided methods as needed (e.g. _SklearnDataMixin or _PytorchModelMixin)
- [x] Audit framework declarations
- [x] Ensure YAML equivalents exist
- [x] Add installed-package detection
- [x] Register only available integrations
- [x] Add runtime parsing support
- [x] Remove declaration Python files
- [x] Add compose tests isolated per-framework (no cross-framework imports)
- [x] Ensure each framework is importable when sibling frameworks are absent

---

# TODO — `deckard/plugins/`

**Goal**: Entirely decouple each plugin family from the abstract core objects and from every other plugin family. Each `deckard/plugins/<family>` must be importable with only its own optional dependency installed. 

Plugin test gate policy:
- This is the phase where plugin-family objects must pass (module blocks only require base-object pass criteria).

- [x] Create new folder one for each 
- [x] Audit plugin files (**anjana.py, **fairlearn.py, **survival.py*, yellowbrick.py, seaborn.py, etc)
- [x] All plugins and their objects should be named after the package-- not what they do to avoid collision (e.g. Survival -> lifelines)
- [x] Refactor all (optional) plugin files to exist in a singular /plugins/<plugin> folder
- [x] Move all plugin definitions to this folder
- [x] Create data.py/model.py/experiment.py/etc as needed.
- [x] Move `AnjanaModelConfig` implementation into `deckard/plugins/anjana/model.py`, remove the legacy `deckard/model/anjana.py` shim, and update source imports to the plugin path
- [x] Move fairlearn model/defense implementations into `deckard/plugins/fairlearn/model.py` and remove legacy `deckard/model/fairness.py` + `deckard/model/fairlearn.py` shims
- [x] Move lifelines survival model implementation into `deckard/plugins/lifelines/model.py` and remove legacy `deckard/model/survival.py` + `deckard/model/lifelines.py` shims
- [x] Migrate remaining plugin-specific implementations out of the deckard core packages into `deckard/plugins/<family>/` and remove core-module copies
- [x] ~~Replace moved core plugin-family modules with compatibility shims (`deckard/data/anjana.py`, `deckard/score/{anjana,fairness,survival}.py`, `deckard/experiment/survival.py`, `deckard/plot/{survival,seaborn_plots,yellowbrick_plots}.py`)~~
- [ ] Remove compatibility shims and fix all broken imports/_target_ references at their call sites
  - [x] Batch 1: removed obsolete `deckard/data/anjana.py` and `deckard/model/anjana.py`; migrated in-repo Anjana `_target_`/score-function paths to `deckard.plugins.anjana.*`
  - [x] Batch 2: migrated remaining fairlearn score call sites in source-adjacent tests/examples from `deckard.plugins.fairlearn.score*` to canonical `deckard.plugins.fairlearn.score*` (`test/test_attack/test_attack.py`, `test/test_frameworks/test_pytorch/test_pytorch_attack.py`, `examples/sklearn/config/score/fairness*.yaml`)
  - [x] Batch 3: migrated legacy plot alias call sites to canonical plugin paths (`deckard.plugins.yellowbrick.plot` -> `deckard.plugins.yellowbrick.plot`, `deckard.plugins.seaborn.plot` -> `deckard.plugins.seaborn.plot`) in runtime layer/tests (`deckard/layers/plot.py`, `test/test_layers/test_plot_layer.py`, `test/test_plugins/test_yellowbrick/test_yellowbrick.py`, `test/test_plugins/test_searborn/test_seaborn.py`)
  - [x] Batch 4: migrated remaining PyTorch fairlearn data alias call site from `deckard.pytorch.fairness_data` to canonical framework path `deckard.frameworks.pytorch.fairness_data` (`test/test_plugins/test_fairlearn/test_fairlearn_pytorch.py`)
  - [x] Batch 5: canonicalized remaining non-alias PyTorch call-site imports and `_target_`/`model_type` paths from `deckard.pytorch.*` to `deckard.frameworks.pytorch.*` across source/examples/tests (`deckard/model/declarations.py`, `examples/pytorch/torch_example.py`, `examples/pytorch/config/data/{torch_cifar10,torch_mnist,fairlearn_celeba}.yaml`, `examples/pytorch/config/data/pipeline/pytorch_pipeline.yaml`, `examples/pytorch/config/model/{default,tinynet}.yaml`, `test/test_layers/test_optimize.py`, `test/test_integration/test_{compose_model_configs,hydra_data_config_integration,hydra_model_config_integration}.py`, `test/test_data/test_data_pipeline_compose.py`, `test/test_plugins/test_anjana/test_pytorch_anjana_integration.py`, `test/test_frameworks/test_pytorch/test_{pytorch_data,pytorch_model,pytorch_serialization,pytorch_experiment}.py`)
  - [x] Batch 6: removed explicit user-facing PyTorch alias-coverage import from `test/test_data/test_data_family_aliases.py` (`deckard.pytorch.data`) and replaced it with canonical framework export parity (`deckard.frameworks.pytorch.data`) to reduce residual alias-surface reliance while preserving config identity behavior
- [x] Update internal imports and `_target_` paths to plugin-owned modules (remove core-shim usage in runtime code paths)
- [ ] Ensure YAML equivalents exist: group/<plugin>-<alias>.yml
- [ ] Add plugin discovery
- [ ] Add external config loading
- [ ] Support `DECKARD_CONFIG_DIRS`
- [ ] Add runtime parsing support
- [ ] Remove plugin declaration Python files
- [ ] Add compose tests isolated per-plugin-family (no cross-family imports)
- [ ] Ensure each plugin family is importable when sibling families are absent
  - [x] Added subprocess isolation import tests for score plugin families (`anjana`, `fairlearn`, `lifelines`) in `test/test_package/test_plugin_family_isolation.py`.

# TODO — `deckard/experiment/`

- [x] Audit experiment declarations
- [x] Ensure YAML equivalents exist
- [x] Add runtime parsing support
- [x] Register dynamically
- [x] Remove declaration Python files
- [ ] Enforce the _Mixin -> Plugin -> Config rule using the new abstract base class and implement overrided methods as needed (e.g. _SklearnExperimentMixin or _LifelinesExperimentlMixin)
- [ ] handles run time composition and execution.
- [ ] Add compose tests
- [ ] Refactor experiment tests
  - [x] Collapse repeated Anjana experiment test setup onto the shared YAML-backed helper

---

# Deletion Phase

After each module is fully migrated:

- [ ] Delete `deckard/**/*declaration*.py`
- [ ] Delete obsolete ConfigStore registration utilities
- [ ] Remove dead imports
- [ ] Remove duplicated test fixture configs

---

# Final Integration Phase

Only after all module TODOs complete:

- [ ] Run compose-only test suite
- [ ] Run unit test suite
- [ ] Run experiment test suite
- [ ] Validate selected sklearn examples (representative set)
- [ ] Validate selected pytorch examples (representative set)
- [ ] Validate plugin loading
- [ ] Validate external config roots
- [ ] Validate optional dependency behavior
- [ ] Validate runtime registration ordering
- [ ] Validate deterministic compose behavior
- [ ] Confirm representative matrix coverage documented in tests

---

# Expected End State

## `deckard/` becomes:

- runtime-driven
- plugin-aware
- config-centric
- Hydra-native
- framework-safe
- plugin-safe
- extensible by anyone

## `examples/*/configs` becomes:

- canonical
- testable
- composable
- externally extensible

## Tests become:

- smaller
- deterministic
- compose-first
- less duplicated
- easier to maintain

# User Questions
~~# User Questions~~

- [Resolved] The agent should handle both plan maintenance and direct implementation edits in `deckard/` and `examples/*/configs`.
- [Resolved] The agent should remain single-file-edit-first but may perform coordinated multi-file refactors when explicitly requested.
- [Resolved] Repository-specific test/validation commands should be included, including config composition/calling/declaration tests and `deckard optimize <hydra overrides>` with blank `default.yaml` context.

---

# Final Tree Diagram

```text
deckard/
├── __init__.py
├── declarations.py
├── frameworks/
│   ├── __init__.py
│   ├── ~~base.py~~ -> core.py
│   ├── sklearn/
│   │   ├── __init__.py
│   │   └── defense.py
│   └── pytorch/
│       ├── __init__.py
│       ├── defense.py
│       ├── experiment.py
│       └── score.py
├── plugins/
│   ├── __init__.py
│   ├── anjana/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── model.py
│   │   └── score.py
│   ├── fairlearn/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── model.py
│   │   └── score.py
│   ├── lifelines/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── experiment.py
│   │   ├── model.py
│   │   ├── plot.py
│   │   └── score.py
│   ├── yellowbrick/
│   │   ├── __init__.py
│   │   └── plot.py
│   └── seaborn/
│       ├── __init__.py
│       └── plot.py
├── data/
│   ├── __init__.py
│   ├── _mixins.py
│   ├── ~~base.py~~ -> core.py
│   └── pipeline/
│       ├── __init__.py
│       ├── core.py
│       ├── ~~default.py~~ -> core.py
│       └── pytorch.py
├── model/
│   ├── __init__.py
│   ├── ~~base.py~~ -> core.py
│   └── defense/
│       
│       ├── ~~default.py~~ -> core.py
│       ├── ~~sklearn.py~~ -> frameworks/sklearn/defense.py
│       └── ~~pytorch.py~~ -> frameworks/pytorch/defense.py
├── attack/      # core attack package
│   ├── __init__.py
│   ├── base.py
│   ├── declarations.py
│   ├── evasion.py
│   ├── extraction.py
│   ├── inference.py
│   ├── poisoning.py
│   ├── pytorch.py
│   ├── reconstruction.py
│   └── torch_utils.py
├── detector/    # core detector package
│   ├── __init__.py
│   ├── base.py
│   └── default.py
├── experiment/  # core experiment package
│   ├── __init__.py
│   ├── base.py
│   ├── declarations.py
│   ├── ~~lifelines.py~~ -> plugins/lifelines/experiment.py
│   ├── ~~pytorch.py~~ -> frameworks/pytorch/experiment.py
│   ├── survival.py
│   └── torch_experiment.py
├── plot/        # core plotting package
│   ├── __init__.py
│   ├── base.py
│   ├── declarations.py
│   ├── ~~lifelines.py~~ -> plugins/lifelines/plot.py
│   ├── ~~seaborn.py~~ -> plugins/seaborn/plot.py
│   ├── seaborn_plots.py
│   ├── survival.py
│   ├── ~~yellowbrick.py~~ -> plugins/yellowbrick/plot.py
│   └── yellowbrick_plots.py
└── score/       # core scoring package
│   ├── __init__.py
│   ├── anjana.py
│   ├── attack.py
│   ├── base.py
│   ├── data.py
│   ├── declarations.py
│   ├── declarations_fairness.py
│   ├── declarations_survival.py
│   ├── ~~fairlearn.py~~ -> plugins/fairlearn/score.py
│   ├── fairness.py
│   ├── ~~lifelines.py~~ -> plugins/lifelines/score.py
│   ├── pytorch.py
│   └── survival.py

examples/
├── sklearn/config/   # canonical Hydra config roots
└── pytorch/config/   # canonical Hydra config roots
```

Legacy plugin-specific files to migrate out of core modules (then delete):

- ~~deckard/data/anjana.py~~ -> `deckard/plugins/anjana/data.py`
- ~~deckard/data/fairness.py~~ -> `deckard/plugins/fairlearn/data.py`
- ~~deckard/data/survival.py~~ -> `deckard/plugins/lifelines/data.py`
- ~~deckard/data/pipeline/anjana.py~~ -> `deckard/plugins/anjana/data.py` (or plugin-specific pipeline module)
- ~~deckard/data/pipeline/fairlearn.py~~ -> `deckard/plugins/fairlearn/data.py` (or plugin-specific pipeline module)
- ~~deckard/model/fairness.py~~ -> `deckard/plugins/fairlearn/model.py`
- ~~deckard/model/survival.py~~ -> `deckard/plugins/lifelines/model.py`
- ~~deckard/plot/seaborn.py~~ -> `deckard/plugins/seaborn/plot.py`
- ~~deckard/plot/seaborn_plots.py~~ -> `deckard/plugins/seaborn/plot.py`
- ~~deckard/plot/yellowbrick_plots.py~~ -> `deckard/plugins/yellowbrick/plot.py`
- ~~deckard/plot/survival.py~~ -> `deckard/plugins/lifelines/plot.py`
- ~~deckard/score/anjana.py~~ -> `deckard/plugins/anjana/score.py`
- ~~deckard/score/fairness.py~~ -> `deckard/plugins/fairlearn/score.py`
- ~~deckard/score/survival.py~~ -> `deckard/plugins/lifelines/score.py`
- ~~deckard/experiment/survival.py~~ -> `deckard/plugins/lifelines/experiment.py`

---

# Phase 4 — Test Consolidation & Minimization

**Goal**: Simplify tests to compose from canonical configs instead of redefining fixtures repeatedly. Implement three-tier testing hierarchy (Compose → Unit → Experiment) with explicit consolidation targets.

---

## Current State Analysis

### Test File Distribution
- **conftest.py**: Root conftest with minimal fixtures (tiny_data, dummy optuna classes)
- **Compose tests**: ~10 files with 50+ test instances (config composition validation)
- **Unit tests**: ~40 files with fixture duplication across modules
- **Integration tests**: ~10 files with 50+ test instances (heavy end-to-end runs)
- **Package tests**: ~5 files (import smoke tests, framework isolation)
- **Total**: ~16 top-level test directories, 100+ test functions

### Key Findings
1. Compose tests exist individually per config profile (e.g., `test_sklearn_data_profile_adult_composes()`)
2. Fixture configs likely duplicated across unit test files
3. Smoke matrix already exists (`test_deckard_optimize_smoke_matrix_sklearn()`)
4. Framework isolation tests are well-structured
5. Fixture generation centralized in root conftest (good re-use pattern)

---

## Three-Tier Testing Strategy

### Level 1 — Compose Tests (Consolidate to 4 files)

**Objective**: Validate configs compose correctly, overrides resolve, defaults are correct.

**Current**: 10+ individual compose test functions
**Target**: 4 parametrized test files (data, model, score, attack)

**Command**: `deckard optimize <hydra overrides> --cfg job`

**Representative Matrix**:
- **Data**: `[sklearn/adult, sklearn/regression, pytorch/mnist, lifelines/rossi]`
- **Model**: `[sklearn/logistic, sklearn/cox, pytorch/resnet]`
- **Score**: `[default, sklearn_classification, pytorch_classification, fairlearn_aware,]`
- **Attack**: `[evasion/fgsm, poisoning/trigger, inference/attribute]`

**TODO**:
- [ ] Create `test_integration/test_compose_data_configs.py` (parametrized)
- [ ] Create `test_integration/test_compose_model_configs.py` (parametrized)
- [ ] Create `test_integration/test_compose_score_configs.py` (parametrized)
- [ ] Create `test_integration/test_compose_attack_configs.py` (parametrized)
- [ ] Verify all compose tests pass
- [ ] Delete legacy `test_hydra_*_config_integration.py` files

### Level 2 — Unit Tests (Consolidate per module)

**Objective**: Validate individual modules (parsers, factories, adapters, scorers) using canonical configs.

**Current**: 40+ test files with fixture duplication
**Target**: 1 consolidated test file per core module with shared YAML-based fixtures

**Strategy**:
- Create module-specific `conftest.py` with shared fixtures (composed configs via Hydra)
- Consolidate related test files per module:
  - `test_data/test_*.py` → `test_data/test_data_unit.py` (parametrized by config)
  - `test_model/test_*.py` → `test_model/test_model_unit.py`
  - `test_attack/test_*.py` → `test_attack/test_attack_unit.py`
  - `test_score/test_*.py` → `test_score/test_score_unit.py`
- Replace Python config definitions with YAML-based test configs

**TODO** (Per module):
- [ ] Create `test_data/conftest.py` with shared data fixtures
- [ ] Consolidate `test_data/*.py` → `test_data/test_data_unit.py`
- [ ] Create `test_model/conftest.py` with shared model fixtures
- [ ] Consolidate `test_model/*.py` → `test_model/test_model_unit.py`
- [ ] Create `test_attack/conftest.py` with shared attack fixtures
- [ ] Consolidate `test_attack/*.py` → `test_attack/test_attack_unit.py`
- [ ] Create `test_score/conftest.py` with shared score fixtures
- [ ] Consolidate `test_score/*.py` → `test_score/test_score_unit.py`
- [ ] Verify all unit tests pass with canonical YAML configs

### Level 3 — Experiment Tests (Minimize to fixed smoke matrix)

**Objective**: Validate end-to-end pipelines with representative paths only (no Cartesian expansion).

**Current**: 50+ integration test functions across 5+ files
**Target**: Single fixed smoke matrix with 7 representative combinations

**Command**: `deckard optimize <hydra overrides>` (run or multirun)

**Fixed Representative Matrix** (NOT parametrized):

| Scenario | Config Overrides | Purpose |
|----------|------------------|---------|
| sklearn baseline evasion | `data=adult model=logistic attack=fgsm score=classification` | Core sklearn path |
| sklearn regression | `data=regression model=ridge attack=none score=regression` | Regression workflow |
| sklearn+fairness | `data=adult model=logistic defense=fairlearn-reductions score=fairlearn-demographic-parity` | Fairness integration |
| pytorch baseline evasion | `data=torch_mnist model=tinynet attack=fgm score=pytorch_classification` | Core pytorch path |
| pytorch+fairness | `data=torch_mnist model=tinynet defense=fairlearn-reductions score=fairlearn-demographic-parity` | PyTorch fairness |
| anjana data pipeline | `data=adult-anjana model=logistic score=classification` | Plugin data path |
| lifelines survival | `data=rossi model=cox score=survival` | Plugin experiment path |

**TODO**:
- [ ] Create `test_integration/test_smoke_matrix.py` (fixed 7-combination matrix)
- [ ] Consolidate `test_anjana_integration.py`, `test_fairness_integration.py`, `test_pytorch_fairness_integration.py`, `test_survival_integration.py` → test_smoke_matrix.py
- [ ] Verify all smoke combinations pass
- [ ] Delete legacy e2e test files (keep framework/base integration tests)

### Level 4 — Plugin-Family Isolation Testing (Deferred)

**Objective**: Test each plugin family in isolation without importing sibling families.

**Target**: Separate test suite per plugin family under `test_plugins/<family>/`

**TODO** (Deferred to plugin consolidation phase):
- [ ] Create `test_plugins/anjana/conftest.py` (anjana-only fixtures)
- [ ] Create `test_plugins/anjana/test_anjana_integration.py` (no sibling imports)
- [ ] Create `test_plugins/fairlearn/conftest.py`
- [ ] Create `test_plugins/fairlearn/test_fairlearn_integration.py`
- [ ] Create `test_plugins/lifelines/conftest.py`
- [ ] Create `test_plugins/lifelines/test_lifelines_integration.py`
- [ ] Verify each family is importable without siblings

---

## Target Test Tree Structure

```text
test/
├── conftest.py                          # Root fixtures (tiny_data, dummy classes)
├── conftest_logging.py
├── helpers.py
│
├── test_package/                        # Package/import validation (UNCHANGED)
│   ├── test_plugin_family_exports.py
│   ├── test_framework_isolation.py
│   ├── test_framework_namespaces.py
│   ├── test_frameworks_package.py
│   ├── test_init_module.py
│   ├── test_classifier_refactor.py
│   ├── test_main.py
│   └── test_plugins_package.py
│
├── test_integration/                    # Consolidated compose + smoke matrix
│   ├── test_compose_data_configs.py     # [NEW] parametrized compose tests
│   ├── test_compose_model_configs.py    # [NEW] parametrized compose tests
│   ├── test_compose_score_configs.py    # [NEW] parametrized compose tests
│   ├── test_compose_attack_configs.py   # [NEW] parametrized compose tests
│   ├── test_smoke_matrix.py             # [NEW] fixed 7-combination experiment matrix
│   ├── test_base_integration.py         # [KEEP] framework base tests
│   ├── test_defense_pipeline_integration.py  # [KEEP] defense pipeline tests
│   └── [DELETE] test_hydra_*_config_integration.py (after consolidation)
│
├── test_data/
│   ├── conftest.py                      # [NEW] data-specific fixtures
│   ├── test_data_unit.py                # [NEW] consolidated data unit tests
│   ├── test_sample.py                   # [KEEP] sample-specific tests
│   ├── test_data_pipeline_*.py          # [KEEP] pipeline-specific tests
│   └── deckard.log
│
├── test_model/
│   ├── conftest.py                      # [NEW] model-specific fixtures
│   ├── test_model_unit.py               # [NEW] consolidated model unit tests
│   ├── test_pytorch.py                  # [KEEP] pytorch-specific (if needed)
│   └── test_pytorch_serialization.py    # [KEEP] serialization-specific
│
├── test_attack/
│   ├── conftest.py                      # [NEW] attack-specific fixtures
│   └── test_attack_unit.py              # [NEW] consolidated attack unit tests
│
├── test_score/
│   ├── conftest.py                      # [NEW] score-specific fixtures
│   └── test_score_unit.py               # [NEW] consolidated score unit tests
│
├── test_experiment/
│   ├── conftest.py                      # [NEW] experiment-specific fixtures
│   └── test_experiment_unit.py          # [NEW] consolidated experiment unit 
├── test_detector/
│   └── test_detector.py                 # [KEEP] detector tests
│
├── test_plot/
│   └── test_plot.py                     # [KEEP] plot tests
│
├── test_layers/
│   └── test_*.py                        # [KEEP] layer-specific tests
│
├── test_utils/
│   └── test_utils.py                    # [KEEP] utility tests
│
├── test_file/
│   └── test_file.py                     # [KEEP] file I/O tests
│
└── test_plugins/                        # [NEW] Plugin-family isolation 
    ├── anjana/
    │   ├── conftest.py
    │   └── test_anjana_integration.py
    ├── fairlearn/
    │   ├── conftest.py
    │   └── test_fairlearn_integration.py
    ├── lifelines/
    │   ├── conftest.py
    │   └── test_lifelines_integration.py
    └── seaborn_yellowbrick/
        ├── conftest.py
        └── test_plotting_integration.py

# Summary:
# - Compose: 4 parametrized files (vs 10+ scattered)
# - Unit: 4 consolidated files (vs 40+ scattered)
# - Experiment: 1 fixed matrix file (vs 5+ scattered)
# - Plugin: 4 family-specific suites (vs 0, deferred)
# - Total reduction: 100+ functions → ~50 core + plugin suite
# - Estimated runtime: 30-40% faster
```

---

## Implementation Checklist

### Phase 4a: Compose Tests (Level 1)
- [x] Create `test_integration/test_compose_data_configs.py`
- [x] Create `test_integration/test_compose_model_configs.py`
- [x] Create `test_integration/test_compose_score_configs.py`
- [x] Create `test_integration/test_compose_attack_configs.py`
- [x] Verify all compose tests pass
- [ ] Archive legacy `test_hydra_*_config_integration.py`

### Phase 4b: Unit Tests per Module (Level 2)
- [ ] Consolidate `test_data/` (create conftest.py, merge test files)
- [ ] Consolidate `test_model/` (create conftest.py, merge test files)
- [x] Consolidate `test_attack/` (create conftest.py, merge test files)
- [x] Consolidate `test_detector/` by merging alias import checks into `test_detector.py`
- [x] Consolidate `test_score/` by merging aliases/context defaults into core score suite
- [ ] Consolidate `test_experiment/` (in progress: aliases merged into core experiment suite)
- [ ] Verify all unit tests pass

### Phase 4c: Experiment Smoke Matrix (Level 3)
- [ ] Create `test_integration/test_smoke_matrix.py` (fixed 7 combinations)
- [ ] Consolidate e2e tests into smoke matrix
- [ ] Verify all smoke combinations pass
- [ ] Archive legacy e2e test files

### Phase 4d: Plugin Isolation Tests (Level 4, Deferred)
- [ ] Create `test_plugins/` directory structure
- [ ] Implement per-family test suites
- [ ] Verify family isolation (no sibling imports)

---

## Consolidation Progress

- [x] Phase 4a: Compose tests consolidated
- [ ] Phase 4b: Unit tests consolidated
- [ ] Phase 4c: Experiment tests consolidated
- [ ] Phase 4d: Plugin tests isolated
- [ ] Final validation: All test layers pass
- [ ] CI/CD updated for new structure

---

## Phase 4a: Compose Tests - COMPLETE ✓

**Completed**: 13 parametrized tests validating canonical config composition.

**Files Created**:
- `test/test_integration/test_compose_data_configs.py` - 5 tests (sklearn: adult, anjana, fairlearn, lifelines; pytorch: torch_mnist)
- `test/test_integration/test_compose_model_configs.py` - 4 tests (sklearn: logistic, cox; pytorch: tinynet; sklearn default override)
- `test/test_integration/test_compose_score_configs.py` - 2 tests (sklearn classification scorers, sklearn survival scorers)
- `test/test_integration/test_compose_attack_configs.py` - 2 tests (sklearn default with attack=hsj, pytorch torch_default with attack=fgm)

**Tests Passing**: 13/13 (0 skipped, 0 failures)

**Key Achievement**: All tests use canonical configs from `examples/sklearn/config` and `examples/pytorch/config` - no mocks, no skips, no missing configs. This validates that canonical configuration declarations are complete and composable.

---

## Phase 4b: Incremental Progress

**Completed in this iteration**:
- Consolidated `test/test_attack/` by merging `test_fairlearn_attack.py` into `test_attack.py`
- Added `test/test_attack/conftest.py` for module-local optional dependency fixtures
- Consolidated `test/test_detector/` by merging alias import checks into `test_detector.py`
- Removed redundant files:
  - `test/test_attack/test_fairlearn_attack.py`
  - `test/test_detector/test_detector_family_aliases.py`
- Updated stale torch utility test imports to canonical path: `deckard.frameworks.pytorch.torch_utils`
- Hardened compose test isolation by resetting Hydra global state and ConfigStore per compose helper

**Validation**:
- `pytest test/test_integration/test_compose_*.py test/test_attack test/test_detector -q`
- Result: **184 passed**

**Additional completed in this iteration**:
- Consolidated `test/test_data/` small duplicate files into `test_data_family_aliases.py`:
  - merged pipeline family import checks
  - merged top-level `PytorchDataPipelineConfig` re-export checks
  - merged default pipeline constructibility checks
- Removed redundant data files:
  - `test/test_data/test_data_pipeline_family_modules.py`
  - `test/test_data/test_data_pipeline_model_imports.py`
  - `test/test_data/test_data_pipeline_package.py`
  - `test/test_data/test_data_pipeline_defaults.py`
- Consolidated `test/test_model/` small duplicate files into `test_model_family_aliases.py`:
  - merged defense package export checks
  - merged default defense baseline marker checks
- Removed redundant model files:
  - `test/test_model/test_model_defense_package.py`
  - `test/test_model/test_model_defense_defaults.py`

**Targeted validation**:
- `pytest test/test_data/test_data_family_aliases.py -q` → **4 passed**
- `pytest test/test_model/test_model_family_aliases.py -q` → **5 passed**
- `pytest test/test_attack test/test_detector test/test_data/test_data_family_aliases.py test/test_model/test_model_family_aliases.py -q` → **180 passed**

**Additional completed in this iteration**:
- Consolidated `test/test_score/` small duplicate files into `test_score.py`:
  - merged score family alias import checks
  - merged YAML profile execution and context-aware default scorer checks
- Removed redundant score files:
  - `test/test_score/test_score_family_aliases.py`
  - `test/test_score/test_score_context_aware_defaults.py`

**Targeted validation updates**:
- `pytest test/test_score/test_score.py -q` → **59 passed**
- `pytest test/test_score -q` → **97 passed**
- `pytest test/test_attack test/test_detector test/test_data/test_data_family_aliases.py test/test_model/test_model_family_aliases.py test/test_score/test_score.py -q` → **239 passed**

**Additional completed in this iteration**:
- Consolidated `test/test_experiment/` alias checks into `test_experiment.py`
- Removed redundant experiment alias file:
  - `test/test_experiment/test_experiment_family_aliases.py`
- Fixed canonical PyTorch data pipeline target path:
  - `examples/pytorch/config/data/pipeline/pytorch_pipeline.yaml`
  - from `deckard.frameworks.pytorch.data.PytorchDataPipelineConfig` to `deckard.pytorch.data.PytorchDataPipelineConfig`

**Experiment validation updates**:
- `pytest test/test_experiment/test_experiment.py -q` → **98 passed, 3 skipped, 1 xfailed**
- `pytest test/test_attack test/test_detector test/test_data/test_data_family_aliases.py test/test_model/test_model_family_aliases.py test/test_score/test_score.py test/test_experiment/test_experiment.py -q` → **337 passed, 3 skipped, 1 xfailed**

**Residual failures outside consolidation scope**:
- Resolved in subsequent iteration; see updates below.

**Subsequent fixes and validation**:
- Added runtime registration opt-out support in `deckard/__init__.py` via env flag:
  - `DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION=1`
- Updated PyTorch experiment subprocess env in `test/test_experiment/test_pytorch_experiment.py` to set:
  - `DECKARD_DEFAULT_CONFIG_FILE=torch_default_cli.yaml`
  - `DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION=1`
- Added `examples/pytorch/config/torch_default_cli.yaml` for CLI subprocess smoke tests
- Canonicalized stale PyTorch YAML targets:
  - `examples/pytorch/config/data/torch_mnist.yaml`
  - `examples/pytorch/config/data/torch_cifar10.yaml`
  - `examples/pytorch/config/data/fairlearn_celeba.yaml`
  - `examples/pytorch/config/data/pipeline/pytorch_pipeline.yaml`
  - `examples/pytorch/config/model/default.yaml`
  - `examples/pytorch/config/model/tinynet.yaml`
- Disabled attack artifact reuse in PyTorch smoke matrix commands by setting:
  - `files.attack_file=null`

**Validation updates**:
- `pytest test/test_experiment/test_pytorch_experiment.py::test_deckard_optimize_torch_art_smoke_matrix -q` → **1 passed**
- `pytest test/test_experiment -q` → **159 passed, 3 skipped, 3 xfailed, 1 xpassed**
- `pytest test/test_attack test/test_detector test/test_data/test_data_family_aliases.py test/test_model/test_model_family_aliases.py test/test_score/test_score.py test/test_experiment/test_experiment.py -q` → **337 passed, 3 skipped, 1 xfailed**

**Additional completed in this iteration**:
- Added focused coverage tests for `deckard/model/detector.py`, `deckard/model/trainer.py`, and `deckard/model/transformer.py`
- Verified file-level coverage improvements:
  - `deckard/model/detector.py` → 100%
  - `deckard/model/trainer.py` → 92%
  - `deckard/model/transformer.py` → 100%
- Added new coverage-oriented tests:
  - `test/test_model/test_model_detector.py`
  - `test/test_model/test_model_trainer.py`
  - `test/test_model/test_model_transformer.py`