# Quickstart

Use this page as the launch point for both first-time users and returning
contributors. It links to the highest-value documentation and explains when to
use each page.

## Start Here

If your goal is to run an experiment quickly:

1. Read [Installation](installation) to create a working environment.
2. Read [Summary](summary) for architectural context.
3. Open the notebook guide in [Notebooks](../notebooks/index).

If your goal is to extend deckard:

1. Read [Developer Docs](development).
2. Review [API](../api/modules).
3. Review extension APIs in [Extensions](extensions).

Core extension docs:

- [PyTorch](../api/pytorch)
- [Fairlearn](../api/fairlearn)
- [Anjana](../api/anjana)
- [Lifelines](../api/lifelines)
- [Seaborn](../api/seaborn)
- [Yellowbrick](../api/yellowbrick)

## Documentation Map

## [Summary](summary)
A conceptual summary of the package.

## [Installation](installation)
Installation instructions for users.

## [API](../api/modules)
Core package documentation

## [Developer Docs](development)
Documentation for testing and extending this package.

## [Docs Docs](build_docs)
Documentation about how to build this documentation.

## [Extensions](extensions)
Extension points and optional subsystems for additional workflows.

## [Changelog](changelog)
A history of changes.

## Recommended Learning Paths

### Path A: Experiment Users

- Install dependencies.
- Run one notebook workflow ([sklearn](../notebooks/sklearn) or [pytorch](../notebooks/pytorch)).
- Inspect scoring outputs and artifacts.
- Adapt one config for a new dataset, model, or metric.

### Path B: Framework Contributors

- Read API module docs for the area you are extending.
- Follow development/testing conventions.
- Add or update docs and notebook examples for new behavior.
- Validate with local workflows and documentation builds.

## Notes On Scope

The pages linked here focus on documentation and architecture orientation.
Executable examples and reproducible runs are covered in notebooks and examples
directories, while module-level behavior is captured in API references.

```{toctree}
:hidden:
summary
installation
development
build_docs
extensions
changelog
```