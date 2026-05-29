# deckard

deckard is a configuration-driven framework for evaluating machine learning
systems under trustworthiness constraints. The docs are organized so the first
things you see are the parts that shape almost every run: configuration,
orchestration, reproducibility, and the trustworthiness metrics that sit on top
of them.

## First Things First

Start here if you want the shortest path through the system:

1. [Deckard orchestration](overview/experiment) for execution flow.
2. [Hydra](overview/hydra) for configuration composition.
3. [Optuna](overview/optimize) for optimization.
4. [DVC](overview/dvc) for reproducibility and artifact tracking.

Trustworthiness concerns come next and are built into the same workflow:

1. [ART / robustness](api/modules) through adversarial attack and defense components.
2. [Anjana](api/anjana) for privacy-aware preprocessing and anonymization.
3. [Fairlearn](api/fairlearn) for fairness-aware evaluation.
4. [Lifelines](api/lifelines) for survival and failure modeling.

Popular framework and plotting support is available through extensions:

1. [sklearn](overview/extensions/sklearn)
2. [PyTorch](overview/extensions/pytorch)
3. [Seaborn](overview/extensions/seaborn)
4. [Yellowbrick](overview/extensions/yellowbrick)

## Where To Go Next

If you are new to the project, begin with:

1. [Overview -> Quickstart](overview/quickstart)
2. [Overview -> Core Modules](overview/core)
3. [Overview -> Experiment Workflow](overview/experiment)
4. [Notebooks -> sklearn](notebooks/sklearn) or [Notebooks -> pytorch](notebooks/pytorch)

If you are extending the framework, begin with:

1. [Core API](api/modules)
2. [Overview -> Extensions](overview/extensions/index)
3. [Developer Docs](developers/index)

## Reference Areas

Use these sections for the deeper material behind the landing page:

- [Overview](overview/index): the main conceptual and workflow entry point.
- [Core API](api/modules): module-level behavior, defaults, and extension points.
- [Extensions](overview/extensions/index): framework and plugin subsystems.
- [Notebooks](notebooks/index): executable end-to-end examples.
- [Developer Docs](developers/index): documentation standards, security notes, and contribution guidance.

## Licensing

- Project license: [GPLv3 License](../LICENSE)
- Dependency and plugin license references: [LICENSES](LICENSES)

```{toctree}
:maxdepth: 2
:caption: Overview
:hidden:

overview/index
LICENSES
```

```{toctree}
:maxdepth: 2
:caption: Core API
:hidden:

api/modules
```

```{toctree}
:maxdepth: 2
:caption: Notebooks
:hidden:

notebooks/index
```

```{toctree}
:maxdepth: 2
:caption: Developer Docs
:hidden:

developers/index
```