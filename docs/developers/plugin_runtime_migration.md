# Plugin Runtime Migration Guardrails

This document defines migration constraints for plugin and framework data
configuration APIs during canon refactors.

## Preserve Top-Level Config APIs

Migration updates must preserve top-level config entry points used by user
configs and Hydra declarations.

Required public paths include:

- `deckard.plugins.fairlearn.FairlearnDataConfig`
- `deckard.plugins.anjana.AnjanaDataConfig`
- `deckard.frameworks.pytorch.data.PytorchDataConfig`
- `deckard.frameworks.pytorch.data.PytorchCustomDataConfig`

Do not migrate user-facing configs to plugin internals such as
`deckard.plugins.<family>.data.<Class>`.

## Hook Bundle Composition

Pipeline and scoring hooks must remain separately named and composable.

Expected pattern:

- pipeline hook bundle: stage hooks such as `before_sample`, `after_pipeline`
- scoring hook bundle: score-tail hooks such as `after_score*`
- composed runtime plugin list from named bundles

## Files-Only Persistence Guard

Data config persistence remains canonical and files-only.

Migration guard checks should reject legacy persistence keys as top-level config
keys in `examples/*/config/data/**/*.yaml`:

- `data_file`
- `score_file`
- `post_sample_data_file`
- `post_pipeline_data_file`

## Score Mode Guard

`score_mode` in configs must be split-scoped only:

- allowed: `train`, `test`, `val`, `all`
- disallowed: stage lifecycle aliases (`post-pipeline`, etc)

## Examples Migration Checklist

When migrating example data profiles:

1. Update `_target_` values away from deprecated paths.
1. Keep plugin targets on top-level plugin APIs.
1. Remove legacy persistence keys from top-level data config payloads.
1. Verify score mode values use split scope only.
1. Run focused migration guard tests.

## Validation Commands

```bash
pytest -q test/test_data/test_contracts.py test/test_data/test_migration_guards.py
pytest -q test/test_plugins/test_fairlearn/test_fairlearn_data_config.py test/test_plugins/test_fairlearn/test_fairness_integration.py test/test_plugins/test_anjana/test_anjana_integration.py
```
