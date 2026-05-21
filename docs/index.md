
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


## Docstring and API Documentation Standard

All public APIs in Deckard use [MyST-native Google-style docstrings](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#google-style-docstrings) rendered via `sphinx.ext.napoleon` and MyST-NB. See the [Docstring Standard](developers/docstring_standard.md) for canonical format, required sections, and syntax rules.

API documentation is generated directly from these docstrings. See [Core API](api/modules) for module-level documentation and [Notebooks](notebooks/index) for executable examples.

If you are new to the project, begin with:

1. [Overview -> Quickstart](overview/quickstart)
2. [Overview -> Summary](overview/summary)
3. [Notebooks -> sklearn](notebooks/sklearn) or [Notebooks -> pytorch](notebooks/pytorch)

If you are extending the framework, begin with:

1. [Core API -> modules](api/modules)
2. [Developer Docs -> development](developers/development)
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
