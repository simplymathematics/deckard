# deckard

deckard is a configuration-driven framework for evaluating machine learning
systems under trustworthiness constraints, including adversarial robustness,
fairness, privacy-aware preprocessing, and failure-aware analysis.

This documentation is organized so you can move from concepts to execution:

- Start in the Overview section for orientation, installation, and workflow
	context.
- Use Core API for module-level behavior and extension points.
- Use Extensions for optional subsystems such as PyTorch integrations and
	visualization stacks.
- Use Notebooks for executable, end-to-end examples.

If you are new to the project, begin with:

1. [Overview -> Quickstart](overview/quickstart)
2. [Overview -> Summary](overview/summary)
3. [Notebooks -> sklearn](notebooks/sklearn) or [Notebooks -> pytorch](notebooks/pytorch)

If you are extending the framework, begin with:

1. [Core API -> modules](api/modules)
2. [Overview -> development](overview/development)
3. [Extensions](overview/extensions)

```{toctree}
:maxdepth: 2
:caption: Overview
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

overview/extensions
```

```{toctree}
:maxdepth: 2
:caption: Notebooks
:hidden:

notebooks/index
```
