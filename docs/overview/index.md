# Overview

## Purpose

The **Overview** section is the fastest way to understand how **deckard** runs
reproducible, multi-objective ML optimization and how to perform post-hoc
analysis from persisted experiment artifacts.

It is intended for:

- **Researchers** who need repeatable experiment workflows
- **Engineers** who need structured security, fairness, and privacy benchmarking
- **Contributors** extending data, model, attack, or scoring components

Core themes covered in this section:

- dependency setup for core and optional extension stacks
- concise composition of base runtime configs
  ([DataConfig](../api/data), [ModelConfig](../api/model),
  [AttackConfig](../api/attack), [DetectorConfig](../api/detector),
  [ExperimentConfig](../api/experiment), [FileConfig](../api/file))
- scoring and persistence as first-class optimization outputs
- multi-objective optimization via [Optuna](https://optuna.org) and
  [Hydra](https://hydra.cc)
- post-hoc evaluation pipelines via [Layers](../api/layers)

Canonical base-config navigation order used across overview docs:

1. data
2. pipeline
3. model
4. trainer
5. defense
6. attack
7. detector
8. scorer
9. files
10. artifacts
11. experiment
12. plot
13. utils
14. framework
15. plugins

## Recommended Reading Order

The following pages are ordered for progressive onboarding:

1. [Quickstart](quickstart.md)
2. [Summary](summary.md)
3. [Optimization](optimize.md)
4. [Hydra](hydra.md)
5. [Data](data.md)
6. [Pipeline](pipeline.md)
7. [Model](model.md)
8. [Trainer](trainer.md)
9. [Defense](defense.md)
10. [Attack](attack.md)
11. [Detector](detector.md)
12. [Scorer (Scoring)](scoring.md)
13. [Files](file.md)
14. [Artifacts](artifacts.md)
15. [Experiment](experiment.md)
16. [Plot](plot.md)
17. [DVC](dvc.md)
18. [Utils](utils.md)
19. [Framework: sklearn](sklearn.md)
20. [Framework: PyTorch](pytorch.md)
21. [Framework: Transformers](transformers.md)
22. [Plugin: ANJANA](anjana.md)
23. [Plugin: Fairlearn](fairlearn.md)
24. [Plugin: Lifelines](lifelines.md)
25. [Plugin: Seaborn](seaborn.md)
26. [Plugin: Yellowbrick](yellowbrick.md)
27. [Extensions](extensions.md)
28. [Installation](installation.md)
29. [Notebooks](../notebooks/index.md)
30. [API Reference](../api/modules.md)
31. [Developer Docs](../developers/index.md)
32. [Development](../developers/index.md)
33. [Build Docs](build_docs.md)
34. [Docker](docker.md)
35. [Changelog](changelog.md)

## Navigation Notes

Each page in this section is designed to be independently useful, but together
they provide a complete map of:

- package architecture
- experiment composition
- reproducibility workflows
- extension and contribution patterns

Use the sidebar for direct navigation, or follow the recommended reading order
for a structured introduction.

```{toctree}
:maxdepth: 2
:hidden:

quickstart
summary
optimize
hydra
data
pipeline
model
trainer
defense
attack
detector
scoring
file
artifacts
experiment
plot
dvc
utils

# Framework overviews
sklearn
pytorch
transformers

# Plugin overviews
extensions
anjana
fairlearn
lifelines
seaborn
yellowbrick

# Software Guide
installation
build_docs
docker
changelog
```
