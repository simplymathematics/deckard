# Utils

## Introduction

This page is the canonical home for shared utility behavior and API details.
It documents deterministic normalization helpers, class/config resolution,
and artifact/persistence support utilities used across runtimes.

The {mod}`deckard.utils` module contains shared utilities used across the
public API, including stable config hashing, serialization helpers, dynamic
class loading, and parser generation helpers.

```{eval-rst}
.. automodule:: deckard.utils
   :members:
   :show-inheritance:
```

## Overview

Utilities provide the shared primitives that keep deckard configs and runtime
behavior deterministic across CLI and programmatic execution.

Key responsibilities include:

- stable hashing for config identity
- safe object/data serialization helpers
- dynamic class loading from import paths
- parser creation from callable signatures
- torch device resolution helpers for cpu/cuda/mps selection
- ConfigStore-safe registration helpers for Hydra config groups

## Examples

```{seealso}

   Utility helpers are exercised throughout the executable notebooks, including:

   - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`

```

## Internals

The module emphasizes deterministic normalization (for hashing and persistence)
and defensive loading behavior so configs are portable across environments.

## Troubleshooting

- Verify dotted import paths when using dynamic class loading helpers.
- Ensure serialized object/data formats match file extension and expected loader.
- Check hash normalization inputs when comparing run identity across platforms.

### See also

- {doc}`../experiment/index`
- {doc}`../file/index`
- {doc}`../layers/index`
