# File

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

## Typical Workflow

1. Configure file outputs through the active experiment config.
1. Execute experiment/model/attack/score layers.
1. Persist and reload artifacts via file config helpers.

Hydra-focused workflow notes:

1. Compose output roots via Hydra config groups and overrides.
1. Resolve run-specific paths after OmegaConf interpolation.
1. Keep artifact paths consistent across {doc}`experiment`, {doc}`plot`, and
   {doc}`layers` stages.

## Troubleshooting

- Ensure output directories are writable.
- Verify artifact paths are consistent across experiment and layer configs.
- Check that expected file formats match the configured save/load behavior.

### See also

- {doc}`experiment` — experiment orchestration
- {doc}`data` — dataset artifacts
- {doc}`model` — model artifacts
- {doc}`score` — score persistence and loading
- {doc}`plot` — plot artifact outputs
- {doc}`layers` — compile/plot/pareto layer outputs
