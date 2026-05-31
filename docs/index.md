# deckard

deckard is a configuration-driven framework for evaluating machine learning
systems under trustworthiness constraints. The docs are organized so the first
things you see are the parts that shape almost every run: configuration,
orchestration, reproducibility, and the trustworthiness metrics that sit on top
of them.

The documentation follows one consistent story:
Overview -> Core API -> Extensions -> Notebooks -> Developer Docs.
Use this page as the routing layer into those sections.

## First Things First

Start here if you want the shortest path through the system:

1. [Deckard orchestration](overview/experiment) for execution flow.
2. [Hydra](overview/hydra) for configuration composition.
3. [Optuna](overview/optimize) for optimization.
4. [DVC](overview/dvc) for reproducibility and artifact tracking.

Trustworthiness concerns come next and are built into the same workflow:

1. [ART / robustness](api/modules) through adversarial attack and defense components. Notebooks: [art_attacks](notebooks/art_attacks), [art_defenses](notebooks/art_defenses), [detector](notebooks/detector).
2. [Anjana](api/plugins/anjana) for privacy-aware preprocessing and anonymization. Notebook: [anjana](notebooks/anjana).
3. [Fairlearn](api/plugins/fairlearn) for fairness-aware evaluation. Notebook: [fairlearn](notebooks/fairlearn).
4. [Lifelines](api/plugins/lifelines) for survival and failure modeling. Notebook: [lifelines](notebooks/lifelines).
5. [TextAttack](api/plugins/textattack) for plugin-backed text attack recipes. Notebook: [art_attacks](notebooks/art_attacks).
6. [OpenAttack](api/plugins/openattack) for plugin-backed text attack integrations. Notebook: [art_attacks](notebooks/art_attacks).

Popular model frameworks is available through framework extensions:

1. [sklearn](overview/extensions/sklearn)
2. [PyTorch](overview/extensions/pytorch)
3. [Transformers](overview/extensions/transformers)

Visualization, diagnostics, and explainability available through plugin extensions.
1. [Seaborn](overview/extensions/seaborn)
2. [Yellowbrick](overview/extensions/yellowbrick)

## Navigation


Choose your own adventure:

### Core Concepts:
	[Overview](overview/index) -> [Quickstart](overview/quickstart) ->
	[Core Modules](overview/core) -> [Experiment Workflow](overview/experiment) ->
	[Notebooks](notebooks/index) -> [sklearn notebook](notebooks/sklearn),
	[pytorch notebook](notebooks/pytorch), or
	[huggingface notebook](notebooks/huggingface)

### CLI and Optimization Workflow:
	[Overview](overview/index) -> [Core Modules](overview/core) ->
	[Hydra](overview/hydra) -> [Optimization](overview/optimize) ->
	[dvc notebook](notebooks/dvc) -> [optuna notebook](notebooks/optuna)

### API + Developer Docs flow (extension and maintenance work):
	[Core API](api/modules) -> [Extensions](overview/extensions/index) ->
	[Developer Docs](developers/index)

## Licensing

- Project license: [GPLv3 License](../LICENSE)
- Dependency and plugin license references: [LICENSES](LICENSES)

```{toctree}
:maxdepth: 2
:caption: Start Here
:hidden:

overview/index
```

```{toctree}
:maxdepth: 2
:caption: Core API
:hidden:

api/modules
```

```{toctree}
:maxdepth: 2
:caption: Extensions
:hidden:

api/plugins/index
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

```{toctree}
:maxdepth: 1
:caption: References
:hidden:

LICENSES
```
