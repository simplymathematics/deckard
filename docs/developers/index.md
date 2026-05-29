# Developer Documentation Index

Welcome to the Deckard developer documentation.
This section contains all design docs, architectural standards, and contributor
guidelines.
All content here is intended for developers extending, maintaining, or
integrating with Deckard.

Use this page as the main contributor entry point for setup, testing,
documentation, and architecture references.

## Docs Map

Use this map to choose where to read first:

- User-facing behavior and usage: {doc}`../overview/index` and {doc}`../api/modules`.
- Developer architecture and runtime contracts: {doc}`design/design`, {doc}`design/orchestration`, {doc}`design/canon_runtime`.
- Extension authoring and execution surfaces: {doc}`extensions/plugins`, {doc}`extensions/hooks`, {doc}`extensions/mixins`.


## Design Docs

The core design pages define the runtime shape, the orchestration contract,
and the canonical execution flow.

- {doc}`Design Principles <design/design>`
- {doc}`Config Declaration Architecture <design/declarations>`
- {doc}`Orchestration Guide <design/orchestration>`
- {doc}`Canon Runtime Execution Guide <design/canon_runtime>`

<!-- Data API -->

## Data API

Data-specific developer docs live with the data runtime contract and sampler
behavior.

- {doc}`Data Design and Contract <data/data>`
- {doc}`Sampler Class Contract <data/samplers>`
- {doc}`Pipeline Class Contract <data/pipelines>`

<!-- Model API -->

## Model API

Model pages cover trainer/defense orchestration, fit/predict semantics, and the
supporting mixin layers.

- {doc}`Model Design and Contract <model/model>`
- {doc}`Trainer Class Contract <model/trainers>`
- {doc}`Defense Class Contract <model/defenses>`

<!-- Attack API -->

## Attack API

Attack and detector docs describe runtime application, scoring, and filtering
behavior.

- {doc}`Attack Design and Contract <attack/attack>`
- {doc}`Detector Design and Contract <attack/detector>`

<!-- Experiment API -->

## Experiment API

Experiment pages tie together the component configs, runtime state, and score
serialization rules.

- {doc}`Experiment Design and Contract <experiment/experiment>`
- {doc}`Score Serialization Contract <experiment/score>`
- {doc}`Plot Design and Contract <experiment/plot>`
- {doc}`Matplotlibrc Behavior and Extension Examples <experiment/matplotlibrc>`

<!-- Persistence API -->

## Persistence API

Persistence docs cover file aliasing, runtime artifacts, and the shared state
helpers used across the pipeline.

- {doc}`Persistence and Runtime State Contract <persistence/persistence>`
- {doc}`DVC Pipeline Autogeneration Spec <optimization/dvc>`
- {doc}`Plugin Runtime Migration Guardrails <contributor/migration>`

<!-- Optimization API -->

## Optimization API

Optimization docs explain Hydra/Optuna wiring and pruning behavior.

- {doc}`Optimization Runtime Contract <optimization/optimization>`
- {doc}`Hydra and Optuna Orchestration Contract <optimization/hydra>`
- {doc}`Pruning Runtime Contract <optimization/pruning>`

<!-- Extension API -->

## Extension API

Extension docs group the shared mixin/plugin rules and the framework/plugin
integration pages.

- {doc}`Developer Extensions <extensions/index>`
- {doc}`Mixin and Plugin Rules <extensions/plugins>`
- {doc}`Plugin and Hook Execution Reference <extensions/hooks>`

<!-- Contributor Notes -->

## Contributor Notes

These pages document the standards and templates used when expanding the
developer docs.

- {doc}`Documentation Standards and Build Guide <contributor/documentation>`
- {doc}`Developer Page Template <contributor/template>`
- {doc}`GH Actions Cache Setup <contributor/actionscache>`
- {doc}`Testing Standards and Escalation Map <contributor/testing>`

## Development Workflow

Deckard is a Python package for declarative AI experimentation, evaluation,
and verification. The project uses:

- [`setuptools`](https://setuptools.pypa.io) build system
- `pyproject.toml` for dependency and tool configuration
- optional extras for modular installs
- CLI entrypoint via `deckard`

### Dependency Model

Core dependencies are defined in `[project].dependencies` in `pyproject.toml`
and are always installed.

Optional dependency groups are defined in `[project.optional-dependencies]`.

Install all optional dependencies:

```bash
pip install ".[all]"
```

Recommended selective installs:

```bash
pip install ".[test]"
pip install ".[docs]"
pip install ".[fairlearn]"
pip install ".[torch]"
```

Other available extras include
[lifelines](../overview/extensions/index),
[anjana](../overview/extensions/index),
[yellowbrick](../overview/extensions/index),
[seaborn](../overview/extensions/index),
as well as HuggingFace `datasets`, and `lint` for code-quality checks.

### Contributor Workflow

Typical contributor loop:

1. Identify the pipeline stage affected ([data](/api/data/index), [model](/api/model/index), [attack](/api/attack/index), [score](/api/score/index),
     or [experiment](/api/experiment/index)).
2. Update implementation and associated declarations/config wiring.
3. Add or update tests for behavior changes.
4. Update notebooks/docs when the user-facing behavior changes.
5. Re-run focused workflows (tests, docs build, notebook stage) before merge.

### Testing and Validation

Contributor testing requirements, fail-fast escalation ordering, and CI mapping
now live in {doc}`contributor/testing`.

### Development Guidelines

- Prefer shared utility helpers over duplicated conversion and normalization logic.
- Keep configuration behavior deterministic and explicit rather than relying on
    implicit fallback behavior.
- Keep metric naming stable, especially for multi-attack and extension metrics,
    to preserve downstream report compatibility.
- When modifying notebook-driven workflows, validate with the corresponding
    [DVC](https://dvc.org) stage.

### Documentation Responsibilities

Code changes that affect behavior should include doc updates in at least one of
the following:

- API page updates when signatures or semantics change.
- Notebook narrative updates when workflow interpretation changes.
- Overview updates when architecture or user entry points change.

### Documentation Build Workflow

Install docs dependencies:

```bash
pip install -e .[docs]
```

Build docs from the `docs/` directory:

```bash
make html
```

For [DVC](https://dvc.org) notebook caching behavior in local and CI docs builds,
see [DVC Cache Setup Summary](contributor/actionscache).

```{toctree}
:maxdepth: 2
:hidden:
:caption: Design Docs

design/design
design/configs
design/declarations
design/orchestration
design/canon_runtime
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Data API

data/data
data/samplers
data/pipelines
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Model API

model/model
model/trainers
model/defenses
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Attack API

attack/attack
attack/detector
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Experiment API

experiment/experiment
experiment/score
experiment/plot
experiment/matplotlibrc
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Persistence API

persistence/persistence
persistence/artifacts
persistence/file
persistence/utils
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Optimization API

optimization/optimization
optimization/hydra
optimization/pruning
optimization/dvc
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Extension API

extensions/mixins
extensions/hooks
extensions/plugins
extensions/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Contributor Notes

contributor/migration
contributor/naming
contributor/documentation
contributor/template
contributor/workflows
contributor/actionscache
contributor/testing
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Future Work

future/refactor_plan
future/security-report
future/prediction-history-implementation-checklist
future/llms
```

For user-facing documentation, see {doc}`../overview/index` and {doc}`../notebooks/index`.
