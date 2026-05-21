# Summary

This chapter introduces the core ideas that shape deckard's architecture and
workflow model.

`deckard` provides a modular framework for defining and executing ML pipelines
in settings where input data can be manipulated, noisy, or adversarial.
Configuration is declarative and YAML-driven, making experiments reproducible
and framework-agnostic while minimizing infrastructure overhead.

The main workflow is multi-objective optimization over configurable scorer
families, followed by post-hoc analysis layers for Pareto filtering,
visualization, and survival diagnostics.

The toolkit includes support for multiple classes of evasion and inference
attacks, as well as fairness-oriented metrics and defenses, and is designed to
be easily extended with additional model, data, metric, and attack components.

Experiment configs support both single-attack and multi-attack workflows.
Multi-attack workflows reuse the same `attack` field by accepting a list of
attack definitions with required aliases for collision-safe metric naming.

In practice, deckard is used both as:

- a backend for large-scale automated evaluation and benchmarking
- a research platform for detailed empirical analysis

## Statement Of Need

Modern ML research often requires combinatorial experimentation across data,
models, evasion/inference attacks, fairness metrics/defenses, and other
evaluation criteria. Managing these experiments manually is error-prone and
difficult to audit.

deckard addresses this by providing:

- configuration-driven orchestration of full ML pipelines around
  multi-objective optimization
- repeatable experiment execution with explicit run metadata
- integrated result collection for comparison and reporting
- [`Hydra`](https://hydra.cc) integration for command line configuration
- [`Optuna`](https://optuna.org) integration for search space sampling,
  experiment pruning, and multi-objective optimization
- **Defense pipelines** that chain [Adversarial Robustness Toolbox
  (ART)](https://adversarial-robustness-toolbox.org/)-based defenses
  (preprocessors, postprocessors, trainers)
- **Detector phase** for auxiliary adversarial/poison detection models executed
  after attacks
- **Data sampling strategies** (split, k-fold, shuffle) for robust evaluation
- **Scorer dictionaries** for unified metric management across data, model, and
  attack components
- Numerous extensions for attacking, defending, and measuring various ML metrics
  (fairness, survival, PyTorch)
- Designed to be easily extensible, but also provide reasonable defaults to
  minimize configuration needs

Base runtime configuration objects used in most experiment graphs:

- [DataConfig](../api/data)
- [ModelConfig](../api/model)
- [AttackConfig](../api/attack)
- [DetectorConfig](../api/detector)
- [ExperimentConfig](../api/experiment)
- [FileConfig](../api/file)

Dependencies are modular: start with core install, then opt into extension
stacks (for example PyTorch, Fairlearn, Lifelines, Seaborn, Yellowbrick,
Anjana) described in [Extensions](extensions) and [Installation](installation).

This reduces engineering friction so researchers can focus on methodology
instead of ad-hoc pipeline glue code.

## Trustworthiness Metrics And Defenses

deckard treats fairness, privacy, and adversarial robustness as first-class
evaluation concerns:

### Fairness And Group-Aware Analysis

Evaluate model behavior across sensitive groups using :mod:`deckard.plugins.fairlearn.data`,
:mod:`deckard.plugins.fairlearn.model`, and
:mod:`deckard.plugins.fairlearn.score`.
These modules
integrate fairlearn for disparate impact, equalized odds, and demographic parity
measurement. Use :class:`~deckard.score.attack.FairlearnAttackScorerConfig` to
measure how adversarial robustness varies across groups.

See [Fairlearn plugin API](../api/fairlearn).

### Privacy And Anonymization

Quantify privacy-utility tradeoffs via :mod:`deckard.Anjana` and
:mod:`deckard.score.anjana`. Configure anonymization strategies (suppression,
bucketing, noise, generalization) and measure information loss, privacy
guarantees, and accuracy impact.

### Adversarial Robustness And Attacks

Execute evasion, membership inference, attribute inference, and model inversion
attacks via :class:`~deckard.attack.AttackConfig` with full [Adversarial
Robustness Toolbox (ART)](https://adversarial-robustness-toolbox.org/)
integration.
Chain defenses using :class:`~deckard.model.DefensePipelineConfig`. Measure
attack success rates, defense effectiveness, and certified robustness bounds.
Combine with fairness analysis for group-aware robustness metrics.

### Survival And Failure Modeling

Model time-to-event outcomes for both raw data processes and ML pipeline
failures using :mod:`deckard.data.survival`,
:mod:`deckard.plugins.lifelines.model`, and
:mod:`deckard.plugins.survival.score`. This supports benign failure analysis (natural
performance degradation or operational failures) and adversarial failure
analysis (attack-induced failure events) within the same reproducible workflow.

### Custom Scorers

All metric types (fairness, privacy, attack, standard) integrate through
:class:`deckard.score.ScorerDictConfig` for unified scoring interface. Define
custom metrics or wrap scikit-learn, numpy, or domain-specific scoring
functions.

These scorer definitions become optimization objectives in Optuna studies and
remain available for downstream layer analysis.

## Usage

### Programmatic Example

```python
from deckard import DataConfig, ModelConfig, AttackConfig, ExperimentConfig

data = DataConfig(dataset_name="adult", test_size=0.2)
model = ModelConfig(model_type="sklearn.linear_model.LogisticRegression")
attack = AttackConfig()
experiment = ExperimentConfig(data=data, model=model, attack=attack)

scores = experiment()
print(scores)
```

### Command-Line Orientation

Use the package through module entrypoints or the top-level CLI router:

```bash
python -m deckard --help
python -m deckard optimize --help
python -m deckard plot --help
```

### Examples And Prior Work

The [examples](https://github.com/simplymathematics/deckard/tree/main/examples)
directory contains reproducible experiment pipelines used for attack/defense
studies, retraining workflows, survival-analysis based evaluations, and
platform/power analyses.

These examples are intended as executable references for adapting deckard to
new model families and experimental questions.

Example coverage includes:

- attack presets across evasion, inference, inversion, and reconstruction
- scorer profiles for classification, regression, fairness, survival, and
  attack-specific metrics
- plotting presets for Yellowbrick diagnostics and Seaborn score visualizations

## Experiment Management

deckard models each stage of an ML evaluation pipeline as configurable objects.
This supports large parameter sweeps while maintaining a stable, auditable
record of what was run.

Typical workflow composition includes:

1. **Dataset Loading And Sampling**: Load data via
   :class:`~deckard.data.DataConfig`, apply data preprocessing pipelines via
   :class:`~deckard.data.DataPipelineConfig`, and sample via pluggable
   :class:`~deckard.data.sample.BaseSampler` strategies.
1. **Preprocessing And Feature Handling**: Transform features via sklearn
   pipelines; automatically instrumented with timing metrics.
1. **Model Training And Evaluation**: Train models via
   :class:`~deckard.model.ModelConfig` with configurable scorer profiles for
   classification, regression, fairness, and survival tasks.
1. **Optional Defenses**: Apply adversarial robustness defenses via
   :class:`~deckard.model.DefensePipelineConfig` that chain ART-based
   preprocessing and postprocessing defenses.
1. **Attack Execution**: Execute evasion or inference attacks via
   :class:`~deckard.attack.AttackConfig` with attack-specific scoring and metric
   aggregation.
1. **Optional Detector Execution**: Train/evaluate auxiliary
   clean-vs-adversarial detectors via :class:`~deckard.detector.DetectorConfig`.
1. **Scoring And Artifact Persistence**: Normalize metrics via
   :class:`~deckard.score.ScorerDictConfig` and persist results via
   :class:`~deckard.file.FileConfig`.

By standardizing these stages, deckard reduces ambiguity in experiment setup
and makes comparative benchmarking easier.

## Optimization-First Workflow

1. Define objective scorers in [Score API](../api/score).
1. Compose experiment config via [Hydra](https://hydra.cc) config groups.
1. Run single or multi-objective optimization through [Optuna](https://optuna.org).
1. Persist predictions, scores, and metadata through [File API](../api/file).
1. Run post-hoc layers for Pareto selection and visual diagnostics via
   [Layers API](../api/layers).

## Reproducibility And Auditability

deckard emphasizes reproducibility by making configuration and artifacts first
class outputs of every run.

- runs are defined by machine-readable config files
- parameters are serialized and can be hashed for run identity
- output artifacts can be tracked in version-controlled workflows

Persistence includes scorer outputs, run metadata, model/data artifacts, and
plot-ready tabular outputs used by post-hoc layers.

This supports internal auditability requirements and reproducible research
publication practices.

## Parallel And Distributed Design

deckard is designed to run on laptops, single servers, and cluster-backed
environments. Through [Hydra](https://hydra.cc)-based composition and optimizer/scheduler-friendly
configuration, the same experiment definition can be reused across:

- local iterative development
- parallel parameter sweeps
- distributed batch execution

This enables scalable trustworthiness studies without rewriting experiment code
for each execution backend, allowing researchers to focus only on the component
that they are truly testing while gaining access to numerous mitigations,
defenses, attacks, and metrics for validating ML pipelines.

Hydra launcher backends provide execution-mode flexibility for these workflows.
Because `joblib` is already a dependency in many deckard environments, the
Joblib launcher is a practical default for local parallel sweeps:

- Joblib launcher:
  [Hydra plugin docs](https://hydra.cc/docs/plugins/joblib_launcher/) |
  [upstream Joblib docs](https://joblib.readthedocs.io/)
- Ray launcher:
  [Hydra plugin docs](https://hydra.cc/docs/plugins/ray_launcher/) |
  [upstream Ray docs](https://docs.ray.io/en/latest/)
- RQ launcher:
  [Hydra plugin docs](https://hydra.cc/docs/plugins/rq_launcher/) |
  [upstream RQ docs](https://python-rq.org/)
- Submitit launcher:
  [Hydra plugin docs](https://hydra.cc/docs/plugins/submitit_launcher/) |
  [upstream Submitit project](https://github.com/facebookincubator/submitit)

See also:

- [Package Summary](summary.md).
- [API Reference](../api/modules)
- [Extensions](extensions.md)
- [Scoring Guide](scoring)
- [Layers](../api/layers)
- [Notebooks](../notebooks/index)
- [Developer Docs](../developers/development)
