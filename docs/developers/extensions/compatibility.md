# Compatibility and Re-exports

## Summary

Deckard optional framework/plugin behavior is registry-driven.

- Source of truth: {mod}`deckard._optional` (`OPTIONAL_FAMILY_REGISTRY`)
- Core modules re-export optional symbols lazily
- Plugin top-level packages expose compatibility aliases lazily

## Optional Families

| Kind | Families |
|---|---|
| framework | `pytorch`, `sklearn`, `transformers_framework` |
| plugin | `anjana`, `fairlearn`, `lifelines`, `openattack`, `seaborn`, `textattack`, `transformers`, `yellowbrick` |

## Core Re-export Surfaces

| Core surface | Optional families |
|---|---|
| `deckard.data` | `anjana`, `fairlearn`, `pytorch` |
| `deckard.model` | `anjana`, `fairlearn`, `lifelines`, `pytorch` |
| `deckard.experiment` | `lifelines`, `pytorch` |
| `deckard.score` | `anjana`, `fairlearn`, `lifelines` |
| `deckard.plot` | `lifelines`, `seaborn`, `yellowbrick` |

## Plugin Alias Surfaces

Compatibility alias packages:

- `deckard.plugins.anjana`
- `deckard.plugins.fairlearn`
- `deckard.plugins.lifelines`

Each package resolves exports through lazy `__getattr__` and dependency gating.

## Rules

- Keep metadata in `OPTIONAL_FAMILY_REGISTRY`.
- Add new optional exports through registry entries, not ad hoc imports.
- Keep compatibility symbol names stable.

## Guardrails

- {mod}`test.test_package.test_plugin_availability_gating`
- {mod}`test.test_package.test_optional_export_loaders`
- {mod}`test.test_data.test_data_family_aliases`
- {mod}`test.test_plot.test_plot_family_aliases`

## See also

- {doc}`/developers/extensions/plugins`
- {doc}`/developers/design/declarations`
- {doc}`/developers/future/refactor_plan`
