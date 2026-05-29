# File

## Introduction

This page is the canonical home for file/artifact path behavior and API
details. It covers file alias contracts, placeholder resolution,
and persistence-oriented runtime helpers.

## Overview

The {mod}`deckard.file` module handles persistence for artifacts produced
throughout deckard runs.

It provides helpers for:

- output path resolution
- score/result serialization
- model and data artifact management
- run directory organization

File outputs are commonly coordinated with [Hydra](https://hydra.cc)
run/multirun directories and [OmegaConf](https://omegaconf.readthedocs.io)
resolved config values.

## Canonical file contract

{class}`deckard.file.FileConfig` is the public typed file registry for canonical artifact paths.
It now uses a shared file handler surface for:

- key validation
- disk-status checks
- placeholder parsing and replacement

Supported placeholders include `{num}`, `{#}`, `{timestamp}`, `{hash}`, and
`{*}`. Hydra job values are used when available, with UUID fallback outside of
Hydra-managed runs.

## Examples

```{seealso}

   Notebook-based file/artifact workflows are documented in:

   - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`

```

## API Reference

```{eval-rst}
.. automodule:: deckard.file
   :members:
   :show-inheritance:
```

## Minimal YAML Example

```yaml
file:
   _target_: deckard.file.FileConfig
   output_dir: build/sklearn
   score_dict_file: score_dict.json
   score_table_file: score_table.csv
```

## Handler Example

{class}`deckard.file.FileConfig` accepts a custom handler when callers need to centralize file-key
validation or disk-status checks.
```python
from deckard.file import CanonFileHandler, FileConfig

handler = CanonFileHandler()
cfg = FileConfig(handler=handler, model_file="build/model.pkl")
status = cfg.disk_status()
```

## Typical Workflow

1. Configure file outputs through the active experiment config.
1. Execute experiment/model/attack/score layers.
1. Persist and reload artifacts via file config helpers.

Hydra-focused workflow notes:

1. Compose output roots via Hydra config groups and overrides.
1. Resolve run-specific paths after OmegaConf interpolation.
1. Keep artifact paths consistent across {doc}`../experiment/index`, {doc}`../plot/index`, and
   {doc}`../layers/index` stages.

## Troubleshooting

- Ensure output directories are writable.
- Verify artifact paths are consistent across experiment and layer configs.
- Check that expected file formats match the configured save/load behavior.

### See also

- {doc}`../experiment/index` — experiment orchestration
- {doc}`../data/index` — dataset artifacts
- {doc}`../model/index` — model artifacts
- {doc}`../score/index` — score persistence and loading
- {doc}`../plot/index` — plot artifact outputs
- {doc}`../layers/index` — compile/plot/pareto layer outputs
