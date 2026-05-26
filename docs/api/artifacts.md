# Artifacts

## Introduction

This page is the canonical home for artifact persistence behavior and API
details. It covers serializer and loader dispatch, canonical artifact payload
handling, and rehydratable runtime outputs.

## Overview

The {mod}`deckard.artifacts` module owns persistence helpers used throughout
deckard runs.

It provides helpers for:

- saving and loading score payloads
- saving and loading data/model/object artifacts
- serializer selection by payload type and suffix
- rehydrating persisted artifacts for rerun and cache-aware workflows

Artifact persistence works alongside {doc}`file` path resolution and the
orchestration behavior documented in {doc}`experiment`.

## Canonical artifact contract

`ArtifactLoaderConfig` owns canonical save/load behavior across score, data,
model, and object payloads.

Typical responsibilities include:

- resolving serializer behavior from payload type and suffix
- preserving human-readable score and params artifacts where possible
- exposing artifacts that can be reloaded in rerun/cache paths

## API Reference

```{eval-rst}
.. automodule:: deckard.artifacts
   :members:
   :show-inheritance:
```

## Minimal YAML Example

```yaml
files:
  _target_: deckard.file.FileConfig
  data_file: outputs/data.pkl
  model_file: outputs/model.pkl
  score_file: outputs/scores.json
```

```yaml
experiment:
  files:
    params_file: outputs/params.yaml
    score_file: outputs/scores.json
```

## Typical Workflow

1. Resolve canonical file aliases via {doc}`file`.
2. Persist runtime payloads through artifact helpers.
3. Reload artifacts for plotting, scoring, reruns, or cache-aware execution.

## See also

- {doc}`file` — canonical path and file alias behavior
- {doc}`experiment` — orchestration and persistence timing
- {doc}`plot` — plot backends consuming persisted artifacts
- {doc}`layers` — post-hoc layers reading persisted outputs
