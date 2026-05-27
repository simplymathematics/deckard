# API Reference

Core module documentation is canonical in this API section.
High-level workflow orientation lives in {doc}`../overview/experiment`.

Developer design goals, constraints, and acceptance criteria live in:

- {doc}`../developers/index`


```



## Shared APIs

Shared user-facing API pages for reusable object families:

```{toctree}
:maxdepth: 1
:caption: Shared APIs

configs
score
```

- Mixin contracts: {doc}`../developers/mixins`
- Plugin class contracts: {doc}`../developers/plugins`

Sub-object API pages remain under their owning trees:

- Data sub-objects: {doc}`sample`, {doc}`pipeline`
- Model sub-objects: {doc}`train`, {doc}`defend`

## Data API

```{toctree}
:maxdepth: 2
:caption: Data

data
sample
pipeline
```

## Model API

```{toctree}
:maxdepth: 2
:caption: Model

model
train
defend
```

## Core Modules

```{toctree}
:maxdepth: 2
:caption: Core Modules

attack
detector
experiment
score
plot
layers
file
artifacts
utils
```

## Framework Integrations

```{toctree}
:maxdepth: 2
:caption: Framework Integrations

pytorch
```

## Plugin Integrations

```{toctree}
:maxdepth: 2
:caption: Plugin Integrations

anjana
fairlearn
lifelines
seaborn
yellowbrick
```
