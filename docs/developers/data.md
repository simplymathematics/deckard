# Data Runtime Canon

This document is the implementation-level contract for Deckard data runtimes
across core, framework, and plugin families.

## Purpose

Define internal data runtime owner contracts for stage hooks, split mode
semantics, persistence invariants, and boundaries between core orchestration,
plugin policy layers, and framework adapters.

## Capabilities

- Define canonical stage and mode semantics and their non-overlapping roles.
- Define hook-orchestrated scoring behavior and stage token normalization.
- Define files-only persistence and canonical timing key guarantees.
- Define plugin and framework-adapter boundaries for data-family extensions.

## Canonical Runtime Contract

Every DataConfig family runtime must preserve these fields and lifecycle methods:

- Runtime attributes: `_X`, `_y`, `X_train`, `X_test`, `X_val`, `y_train`,
  `y_test`, `y_val`, `files`, `times`, `score_dict`
- Lifecycle methods: `load_dataset`, `fit`, `sample`, `score`, `__call__`
- Canonical timing keys in `times`: `data_load_time`, `data_sample_time`,
  `data_pipeline_time`, `data_score_time`

The core helper APIs are implemented in:

- {mod}`deckard.data.canon`
- {mod}`deckard.data.base`
- {mod}`deckard.plugins.base`

## Stage vs Scope Semantics

Deckard separates stage lifecycle from score scope:

- Stage lifecycle (hooks): `pre-load`, `pre-sample`, `post-sample`,
  `post-pipeline`
- Score scope (mode): `train`, `test`, `val`, `all`

`score_mode` is always split-scoped. Stage names are not valid score modes.

Sampling controls are always sampler-scoped. Configure
`train_size`/`test_size`/`val_size` plus `random_state` and `stratify` under
`data.sampler`.

## Files-Only Persistence

Data runtime persistence is files-only via the canonical `files` mapping.

Allowed aliases include:

- `data_file`
- `score_file`
- `metadata_file`
- `post_sample_data_file`
- `post_pipeline_data_file`

Top-level kwargs such as `data_file=...` and `score_file=...` are legacy and
must be normalized into `files={...}` at wrapper boundaries only.

## Hook-Orchestrated Scoring

Score execution is hook-orchestrated through stage-scoped plugin hooks.

Canonical score hook families:

- Before-stage score hooks: `before_score_<stage_token>`
- After-stage score hooks: `after_score_<stage_token>`
- Generic score hooks: `before_score`, `after_score`

Where `<stage_token>` maps from canonical stage names (`post_pipeline`, etc).

## Plugin Policy Layer Rule

Plugin data configs must be policy layers on top of canonical runtime behavior.
They should only define plugin-specific concerns:

- sensitive-feature handling
- mitigation transforms
- plugin scoring policies
- plugin-specific validation

They should not replace canonical data orchestration.

## Cross-Family Guardrail Tests

The contract and migration tests added during this refactor prevent regressions:

- cross-family runtime contract tests (core/framework/plugin)
- stage and hook conformance tests
- example config migration guards

See:

- `test/test_data/test_contracts.py`
- `test/test_data/test_migration_guards.py`
