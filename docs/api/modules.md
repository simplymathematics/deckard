# API Reference

Core module documentation is canonical in this API section.
High-level workflow orientation lives in {doc}`../overview/experiment`.

Developer design goals, constraints, and acceptance criteria live in:

- {doc}`../developers/index`

Start with the core runtime pages, then move through orchestration and
persistence, and finish with the framework, plugin, and CLI entry points.

## Data API

Begin with data, because it establishes the dataset contract that the rest of
the runtime builds on.

```{toctree}
:maxdepth: 2
:caption: Data

data/index
data/sample
data/pipeline
```

## Model API

Model pages come next because they define fit, predict, and defense behavior on
top of the data contract.

```{toctree}
:maxdepth: 2
:caption: Model

model/index
model/train
model/defend
```

## Attack API

Attack and detector pages follow model behavior so you can see how adversarial
and filtering paths layer onto the core runtime.

```{toctree}
:maxdepth: 2
:caption: Attack API

attack/index
detector/index
```
## Experiment API

Experiment pages tie the data, model, attack, detector, and score pieces into a
single runtime flow.

```{toctree}
:maxdepth: 2
:caption: Experiment API
experiment/index
score/index
plot/index


```

## Persistence API

Persistence pages describe the runtime artifacts and file helpers that keep the
experiment flow reproducible.

```{toctree}
:maxdepth: 2
:caption: Persistence API

file/index
artifacts/index
utils/index
```

## Framework Integrations

Framework integrations extend the core runtime with backend-specific behavior,
starting with PyTorch.

```{toctree}
:maxdepth: 2
:caption: Framework Integrations

frameworks/index
```

## Plugin Integrations

Plugin integrations sit after the framework adapters and cover the optional
extension families that layer on fairness, anonymization, and visualization.

```{toctree}
:maxdepth: 2
:caption: Plugin Integrations

plugins/index
```

## Command Line Interface

The CLI section closes the loop with the entrypoints used to run the same flow
from the shell.

```{toctree}
:maxdepth: 2
:caption: CLI

layers/index
