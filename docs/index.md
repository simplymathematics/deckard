# deckard

deckard is a configuration-driven framework for evaluating machine learning
systems under trustworthiness constraints, including adversarial robustness,
fairness, privacy-aware preprocessing, and failure-aware analysis.

This documentation is organized so you can move from concepts to execution:

- Start in the Overview section for orientation and high-level experiment
  workflow context.
- Use Core API for module-level behavior, defaults, and extension points.
- Use Extensions for optional framework and plugin subsystems such as PyTorch,
  Fairlearn, Anjana, Lifelines, Seaborn, and Yellowbrick integrations.
- Use Notebooks for executable, end-to-end examples.

API documentation is generated directly from these docstrings.
See [Core API](api/modules) for module-level documentation,
[Notebooks](notebooks/index) for executable examples, and
[Developer Docs -> Docstrings](developers/docstrings) for the canonical
docstring standard.
Security status and workflow hardening notes are documented in
[Developer Docs -> Security Report](developers/security-report).

If you are new to the project, begin with:

1. [Overview -> Quickstart](overview/quickstart)
2. [Overview -> Summary](overview/summary)
3. [Overview -> Experiment Workflow](overview/experiment)
4. [Notebooks -> sklearn](notebooks/sklearn) or [Notebooks -> pytorch](notebooks/pytorch)

If you are extending the framework, begin with:

1. [Core API -> modules](api/modules)
2. [Developer Docs](developers/index)
3. [Extensions](overview/extensions/index)

## Licensing 

- Project license: [GPLv3 License](../LICENSE)
- Dependency and plugin license references: [LICENSES](LICENSES)



## Frameworks and Plugins

Deckard ships with a small core runtime and a broader extension ecosystem.

Framework integrations:

- [PyTorch API](api/pytorch) covers torch-native models, training flows, and
  experiment execution.

Plugin integrations:

- [Fairlearn API](api/fairlearn) covers fairness-aware workflows and group
  scoring.
- [Anjana API](api/anjana) covers anonymization-aware preprocessing and
  privacy-oriented experimentation.
- [Lifelines API](api/lifelines) covers survival analysis and time-to-event
  workflows.
- [Seaborn API](api/seaborn) and [Yellowbrick API](api/yellowbrick) cover
  visualization, diagnostics, and reporting extensions.

For a fuller extension map, see [Overview -> Extensions](overview/extensions/index).

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
