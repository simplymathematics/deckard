# Score Serialization Contract

This document defines the canonical ScoreDict contract implemented in deckard/artifacts.py.

## Goals

- Use one native score container across runtimes.
- Keep score persistence human-readable while preserving machine-friendly projections.
- Support scalar, vector, and nested score payloads.
- Centralize score load/merge/save behavior.

## Canonical Container

Score payloads are represented by ScoreDict.

Key behaviors:

- Normalization
  - Converts numpy scalars/arrays and pandas objects into JSON/YAML-safe values.
- Nested updates
  - Supports stage/mode/split-aware updates via update_score.
- Views
  - flatten returns dot-delimited key/value mappings.
  - flat_by_scope groups flattened keys by top-level scope.
  - dotlist_dict and dotlist_items expose OmegaConf-friendly formats.
- Contract envelope
  - to_contract_envelope returns schema-tagged payload + flat projections.

## Callable Lifecycle

ScoreDict is callable and owns score persistence lifecycle when a score file is supplied:

- Input: score_file, artifact_loader, persist.
- Behavior:
  - If score_file is absent: returns in-memory nested payload.
  - If score_file exists: loads persisted scores via artifact_loader.load_scores and merges with in-memory scores.
  - If persist is true: writes merged scores via artifact_loader.save_scores.
- Output: nested runtime dictionary.

This replaces legacy read_or_initialize_scores and general merge/persist score flow in BaseConfig.

## Persistence Schema

Schema tag: deckard.score.v1

Persisted envelope fields:

- _schema
- payload
- flat
- flat_by_scope
- dotlist
- dotlist_items

Compatibility behavior:

- payload keys are also mirrored at top-level in serialized JSON/YAML to support older readers that expect direct root keys.

## Runtime Integration

- BaseConfig types score_dict as ScoreDict.
- BaseConfig read_or_initialize_scores and merge_and_persist_scores delegate to ScoreDict callable lifecycle.
- Plugin and runtime score merges normalize score_dict through ScoreDict.from_payload.

## Hooks, Defaults, Plugins, Extensions

This section maps the score API model to runtime internals for maintainers.

- Hooks
  - Score execution supports before/after stage hook orchestration in plugin
    runtimes.
  - Hook outputs that are dict-like are merged into ScoreDict-normalized
    runtime score payloads.
- Defaults
  - Core default scorer families are defined through score config objects in
    deckard.score and consumed by DataConfig/ModelConfig/AttackConfig/
    DetectorConfig paths.
  - Defaults are config-driven, so runtime behavior should avoid hard-coded
    metric logic outside scorer config classes.
- Optional plugins
  - Fairlearn, lifelines, and anjana scorer defaults are optional-dependency
    gated and imported lazily.
  - Contract-safe behavior requires missing-optional paths to fail gracefully
    without breaking core score imports.
- Extensions
  - Preferred extension is ScorerConfig/ScorerDictConfig composition plus
    plugin hook integration.
  - Extended scorers should return scalar or dict-like payloads that remain
    serializable after ScoreDict normalization.

## DVC System Scoring Hook

Deckard includes a DVC-specific scorer profile in
{class}`~deckard.score.DVCSystemScorerDictConfig` for component-scoped runtime
system stats.

Runtime behavior:

- Hook entrypoint: `deckard.experiment.dvc.run_dvc_experiment_plugin_hook`
- Trigger: all `after` hook events at `plugin_position=last`; default scorer
  stage filters execute metrics at `data-score`, `model-score`,
  `attack-score`, and `detector-score`
- Merge target: experiment-level `score_dict`
- Metric naming: concise `<component>_<stat>` keys (for example `data_cpu`,
  `model_memory`, `attack_gpu_power`, `defense_gpu_power`)
- Persistence scope: `score_dict[stage][mode][metric]`
- Source metrics: normalized DVCLive `system_monitor/*` signals plus existing
  power hook metrics (`power/<namespace>/*`) when available

Maintain this contract when extending DVC score hooks:

- Keep payloads scalar/dict-like and JSON/YAML-serializable.
- Keep metric keys concise and component-scoped.
- Preserve stage+mode score scoping when merging runtime hook outputs.

## Implementation Notes

- ScoreDict currently lives in deckard.artifacts to avoid duplicate
  transformation logic across persistence/runtime layers.
- Persisted envelope includes compatibility root-level payload keys for older
  readers, while canonical consumers should use payload/flat/dotlist fields.
- When modifying score merge semantics, update both contract tests and notebook
  examples to keep docs and runtime behavior aligned.

## Notes on y_true/y_pred Migration

Data-scoring paths should use X/y runtime semantics.
Legacy y_true/y_pred usage may still appear in non-data scoring domains (for example attack/model scorers), where it remains valid.
