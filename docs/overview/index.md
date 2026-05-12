# Overview

The Overview section is the fastest way to understand what deckard is, what
problems it solves, and how to run reproducible evaluations with minimal
boilerplate. It is intended for three audiences:

- Researchers who need repeatable experiment workflows.
- Engineers who need structured security/fairness/privacy benchmarking.
- Contributors extending data, model, attack, or scoring components.

Recommended reading order:

1. `quickstart` for practical onboarding.
2. `summary` for architecture and conceptual framing.
3. `installation` to set up local and CI-compatible environments.
4. `development` if you plan to contribute code or docs.
5. `build_docs` for Sphinx and notebook documentation workflows.

Each page in this section is designed to be independently useful, but together
they provide a complete map of how the package is structured and how to reason
about experiment composition.

```{toctree}
:maxdepth: 2
:hidden:

quickstart
summary
installation
development
build_docs
docker
extensions
changelog
```