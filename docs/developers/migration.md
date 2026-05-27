# Plugin Runtime Migration Guardrails

This document defines migration constraints for plugin and framework data
configuration APIs during canon refactors.

## Preserve Top-Level Config APIs

Migration updates must preserve top-level config entry points used by user
configs and Hydra declarations.

Required public paths include:

- {class}`deckard.plugins.fairlearn.FairlearnDataConfig`
- {class}`deckard.plugins.fairlearn.data.FairlearnDataConfig`
- {class}`deckard.plugins.fairlearn.FairnessBehaviorMixin`
- {class}`deckard.plugins.fairlearn.data.FairnessBehaviorMixin`
- {class}`deckard.plugins.anjana.AnjanaDataConfig`
- {class}`deckard.plugins.anjana.data.AnjanaDataConfig`
- {class}`deckard.plugins.anjana.PrivacyBehaviorMixin`
- {class}`deckard.plugins.anjana.data.PrivacyBehaviorMixin`
- {class}`deckard.plugins.lifelines.SurvivalExperimentConfig`
- {class}`deckard.plugins.lifelines.experiment.SurvivalExperimentConfig`
- {class}`deckard.frameworks.pytorch.data.PytorchDataConfig`
- {class}`deckard.frameworks.pytorch.data.PytorchCustomDataConfig`

Compatibility re-exports remain supported at {class}`deckard.data.AnjanaDataConfig` and
{class}`deckard.data.FairlearnDataConfig`, but only when the matching optional plugin
dependencies are installed. Plugin-family packages keep their top-level
aliases stable and lazily resolve them from the owning module so that optional
dependency checks control availability.

Do not migrate user-facing configs to plugin internals such as
`deckard.plugins.<family>.data.<Class>`.

The supported exception is compatibility-only alias imports listed above. Those
aliases should continue to resolve to the canonical module definitions and are
the only plugin-internal import paths that remain part of the migration guard.

## Hook Bundle Composition

Pipeline and scoring hooks must remain separately named and composable.

Expected pattern:

- pipeline hook bundle: stage hooks such as `before_sample`, `after_pipeline`
- scoring hook bundle: score-tail hooks such as `after_score`
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

## Sampler Parameter Guard

Sampling parameters are sampler-owned and must not be configured as top-level
data config keys.

- allowed location: `data.sampler.*`
- sampler-owned fields: `split`, `train_size`, `test_size`, `val_size`,
  `random_state`, `stratify`

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