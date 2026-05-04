Package Overview
================

The :mod:`deckard` package is a declarative toolkit for machine learning
evaluations. It is designed to standardize end-to-end experiments
that combine dataset preparation, model training, scoring, and adversarial
analysis under one reproducible workflow.

.. automodule:: deckard
   :members:
   :no-index:
   :show-inheritance:

Summary
-------

Deckard provides a modular framework for defining and executing ML pipelines
in settings where input data can be manipulated, noisy, or adversarial.
Configuration is declarative and YAML-driven, making experiments reproducible
and framework-agnostic while minimizing infrastructure overhead.

The toolkit includes support for multiple classes of evasion and inference
attacks, as well as fairness-oriented metrics and defenses, and is designed to
be easily extended with additional model, data, metric, and attack components.

In practice, Deckard is used both as:

- a backend for large-scale automated evaluation and benchmarking
- a research platform for detailed empirical analysis



Statement Of Need
-----------------

Modern ML research often requires combinatorial experimentation across data,
models, evasion/inference attacks, fairness metrics/defenses, and other
evaluation criteria. Managing these experiments manually is error-prone and
difficult to audit.

Deckard addresses this by providing:

- configuration-driven orchestration of full ML pipelines
- repeatable experiment execution with explicit run metadata
- integrated result collection for comparison and reporting
- ``hydra`` integration for command line configuration
- ``optuna`` integration for search space sampling, experiment pruning, and multi-objective optimization.
- **Defense pipelines** that chain ART-based defenses (preprocessors, postprocessors, trainers).
- **Detector phase** for auxiliary adversarial/poison detection models executed after attacks.
- **Data sampling strategies** (split, k-fold, shuffle) for robust evaluation.
- **Scorer dictionaries** for unified metric management across data, model, and attack components.
- Numerous extensions for attacking, defending, and measuring various ML metrics (fairness, survival, PyTorch).
- Designed to be easily extensible, but also provide reasonable defaults to minimize configuration needs.

This reduces engineering friction so researchers can focus on methodology
instead of ad-hoc pipeline glue code.

Trustworthiness Metrics & Defenses
----------------------------------

Deckard treats fairness, privacy, and adversarial robustness as first-class evaluation concerns:

**Fairness & Group-Aware Analysis**:

Evaluate model behavior across sensitive groups using :mod:`deckard.data.fairness`,
:mod:`deckard.model.fairness`, and :mod:`deckard.score.fairness`. These modules
integrate fairlearn for disparate impact, equalized odds, and demographic parity
measurement. Use :class:`~deckard.score.attack.FairlearnAttackScorerConfig` to
measure how adversarial robustness varies across groups.

**Privacy & Anonymization**:

Quantify privacy-utility tradeoffs via :mod:`deckard.data.anjana` and
:mod:`deckard.score.anjana`. Configure anonymization strategies (suppression,
bucketing, noise, generalization) and measure information loss, privacy
guarantees, and accuracy impact.

**Adversarial Robustness & Attacks**:

Execute evasion, membership inference, attribute inference, and model inversion attacks via
:class:`~deckard.attack.AttackConfig` with full ART integration. Chain defenses
using :class:`~deckard.model.DefensePipelineConfig`. Measure attack success
rates, defense effectiveness, and certified robustness bounds. Combine with
fairness analysis for group-aware robustness metrics.

**Survival & Failure Modeling**:

Model time-to-event outcomes for both raw data processes and ML pipeline
failures using :mod:`deckard.data.survival`, :mod:`deckard.model.survival`, and
:mod:`deckard.score.survival`. This supports benign failure analysis (natural
performance degradation or operational failures) and adversarial failure
analysis (attack-induced failure events) within the same reproducible workflow.

**Custom Scorers**:

All metric types (fairness, privacy, attack, standard) integrate through
:class:`deckard.score.ScorerDictConfig` for unified scoring interface. Define
custom metrics or wrap scikit-learn, numpy, or domain-specific scoring functions.


Usage
-----

Programmatic Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deckard import DataConfig, ModelConfig, AttackConfig, ExperimentConfig

   data = DataConfig(dataset_name="adult", test_size=0.2)
   model = ModelConfig(model_type="sklearn.linear_model.LogisticRegression")
   attack = AttackConfig()
   experiment = ExperimentConfig(data=data, model=model, attack=attack)

   scores = experiment()
   print(scores)

Command-Line Orientation
~~~~~~~~~~~~~~~~~~~~~~~~

Use the package through module entrypoints or the top-level CLI router:

.. code-block:: bash

   python -m deckard --help
   python -m deckard optimize --help
   python -m deckard plot --help

Examples And Prior Work
~~~~~~~~~~~~~~~~~~~~~~~

The ``examples/`` directory contains reproducible experiment
pipelines used for attack/defense studies, retraining workflows, survival-analysis
based evaluations, and platform/power analyses.

These examples are intended as executable references for adapting Deckard to
new model families and experimental questions.

Experiment Management
---------------------

Deckard models each stage of an ML evaluation pipeline as configurable objects.
This supports large parameter sweeps while maintaining a stable, auditable
record of what was run.

Typical workflow composition includes:

1. **Dataset Loading & Sampling** — Load data via :class:`~deckard.data.DataConfig`, apply data preprocessing pipelines via :class:`~deckard.data.DataPipelineConfig`, and sample via pluggable :class:`~deckard.data.sample.BaseSampler` strategies.
2. **Preprocessing & Feature Handling** — Transform features via sklearn pipelines; automatically instrumented with timing metrics.
3. **Model Training & Evaluation** — Train models via :class:`~deckard.model.ModelConfig` with configurable scorer profiles for classification, regression, fairness, and survival tasks.
4. **Optional Defenses** — Apply adversarial robustness defenses via :class:`~deckard.model.DefensePipelineConfig` that chain ART-based preprocessing and postprocessing defenses.
5. **Attack Execution** — Execute evasion or inference attacks via :class:`~deckard.attack.AttackConfig` with attack-specific scoring and metric aggregation.
6. **Optional Detector Execution** — Train/evaluate auxiliary clean-vs-adversarial detectors via :class:`~deckard.detector.DetectorConfig`.
7. **Scoring & Artifact Persistence** — Normalize metrics via :class:`~deckard.score.ScorerDictConfig` and persist results via :class:`~deckard.file.FileConfig`.

By standardizing these stages, Deckard reduces ambiguity in experiment setup and makes comparative benchmarking easier.

Reproducibility And Auditability
--------------------------------

Deckard emphasizes reproducibility by making configuration and artifacts first
class outputs of every run.

- runs are defined by machine-readable config files
- parameters are serialized and can be hashed for run identity
- output artifacts can be tracked in version-controlled workflows

This supports internal auditability requirements and reproducible research
publication practices.

Parallel And Distributed Design
-------------------------------

Deckard is designed to run on laptops, single servers, and cluster-backed
environments. Through Hydra-based composition and optimizer/scheduler-friendly
configuration, the same experiment definition can be reused across:

- local iterative development
- parallel parameter sweeps
- distributed batch execution

This enables scalable trustworthiness studies without rewriting experiment code for
each execution backend, allowing researchers to focus only on the component that they are truly testing while gaining access to numerous mitigations, defenses, attacks, and metrics for validating ML pipelines.

Internals And Architecture
--------------------------

The package-level API exposes core configuration classes that drive the
workflow: Data, Model, Attack, Experiment, File, and ScorerDict objects.

**Core Config Classes**:

- :class:`deckard.data.DataConfig` — loads, preprocesses, and splits datasets
- :class:`deckard.data.DataPipelineConfig` — configurable sklearn pipelines for feature preprocessing
- :class:`deckard.model.ModelConfig` — instantiates and trains scikit-learn models
- :class:`deckard.model.DefensePipelineConfig` — chains ART defenses (preprocessors, postprocessors, trainers)
- :class:`deckard.detector.DetectorConfig` — auxiliary detector training/evaluation phase
- :class:`deckard.attack.AttackConfig` — executes ART evasion and inference attacks
- :class:`deckard.experiment.ExperimentConfig` — orchestrates end-to-end workflows
- :class:`deckard.file.FileConfig` — manages result serialization and artifact tracking
- :class:`deckard.score.ScorerDictConfig` — unified metric configuration and aggregation

**Defense Architecture**:

Deckard supports `Adversarial Robustness Toolbox (ART) <https://github.com/Trusted-AI/adversarial-robustness-toolbox>`_ defenses by wrapping scikit-learn models as ART estimators. :class:`deckard.model.DefensePipelineConfig` composes multiple defenses into a chain, accumulating preprocessors and postprocessors into a single ART wrapper for efficient ensemble application.

**Scoring Architecture**:

:class:`deckard.score.ScorerDictConfig` normalizes metric definitions into callable maps, supporting classification, regression, fairness, and attack-specific scoring. Attack scoring routes outputs through attack-kind-aware profiles (evasion, membership, attribute, model inversion, database reconstruction) and prefixes metric names automatically.

**Sampling Architecture**:

:class:`deckard.data.sample.BaseSampler` and subclasses provide pluggable train/test/validation split strategies, enabling cross-validation, repeated random splits, and stratified sampling.

**Data Pipeline Architecture**:

:class:`deckard.data.DataPipelineConfig` wraps sklearn's :class:`~sklearn.pipeline.Pipeline` with timing instrumentation and optional normalization. Empty pipelines skip fitting entirely for efficiency.



Extensions And Optional Backends
----------------------------------

Deckard maintains a modular, plugin-based architecture using Hydra's ``ConfigStore``.

**Data Extensions**:

- :mod:`deckard.data.fairness` — :class:`~deckard.data.fairness.FairlearnDataConfig` for group-aware sampling and fairness metrics (fairlearn integration)
- :mod:`deckard.data.anjana` — :class:`~deckard.data.anjana.AnjanaDataConfig` for anonymization-aware data workflows
- :mod:`deckard.data.pytorch` — :class:`~deckard.data.pytorch.PytorchDataConfig` for PyTorch dataset and DataLoader integration
- :mod:`deckard.data.survival` — :class:`~deckard.data.survival.LifelinesDataConfig` for survival analysis with lifelines datasets and auxiliary models
- :mod:`deckard.data.sample` — :class:`~deckard.data.sample.SplitSampler`, :class:`~deckard.data.sample.KFoldSampler`, :class:`~deckard.data.sample.ShuffleSampler` for robust sampling strategies

**Model Extensions**:

- :mod:`deckard.model.fairness` — :class:`~deckard.model.fairness.FairlearnModelConfig` for sklearn fairness-aware model fitting and :class:`~deckard.model.fairness.FairlearnPytorchModelConfig` for PyTorch fairness models with fairlearn defense wrappers
- :mod:`deckard.model.anjana` — :class:`~deckard.model.anjana.AnjanaModelConfig` for anonymization-aware model paths
- :mod:`deckard.model.pytorch` — :class:`~deckard.model.pytorch.PytorchModelConfig` for PyTorch-native model training and prediction
- :mod:`deckard.model.survival` — :class:`~deckard.model.survival.SurvivalModelConfig` for survival models with lifelines estimators
- :mod:`deckard.model.defend` — :class:`~deckard.model.defend.DefenseConfig` and :class:`~deckard.model.defend.DefensePipelineConfig` for ART defense application

**Scoring Extensions**:

- :mod:`deckard.score.fairness` — fairness metrics via :class:`~deckard.score.fairness.FairlearnScoreDictConfig` and default fairlearn scorer profiles
- :mod:`deckard.score.anjana` — anonymization metrics via :class:`~deckard.score.anjana.DefaultAnjanaDataScoreConfig` and :class:`~deckard.score.anjana.DefaultAnjanaModelScoreConfig`
- :mod:`deckard.score.survival` — survival metrics (concordance, AIC, BIC) via :class:`~deckard.score.survival.DefaultLifelinesConfig`
- :mod:`deckard.score.attack` — attack-specific metrics with attack kind-aware routing via :class:`~deckard.score.attack.AttackScorerConfig`
- :mod:`deckard.score.attack` — fairness-stratified attack scoring via :class:`~deckard.score.attack.FairlearnAttackScorerConfig`
- :mod:`deckard.score.data` — data inspection metrics (distributions, imbalance) via :class:`~deckard.score.data.DefaultDataClassificationConfig` and :class:`~deckard.score.data.DefaultDataRegressionConfig`

**Other Extensions**:

- :mod:`deckard.attack` — evasion, membership inference, and attribute inference attacks
- :mod:`deckard.plot` — visualization (seaborn, yellowbrick, survival curves)
- :mod:`deckard.layers` — advanced workflows (Optuna integration, multi-run optimization)
- :mod:`deckard.experiment.torch_experiment` — PyTorch-focused experiment orchestration
- :mod:`deckard.experiment.survival` — survival-specific experiment orchestration

Troubleshooting
---------------

- If interpolations fail, verify resolver inputs and config paths.
- If optional backends are unavailable, install the corresponding extras.
- If behavior differs between runs, compare resolved configs and hash values.

See also
~~~~~~~~

* :doc:`modules` — complete API documentation tree
* :doc:`pytorch` — PyTorch integration guide
* :doc:`anjana` — anonymization support
* :doc:`lifelines` — survival analysis
* :doc:`seaborn` — statistical visualization
* :doc:`yellowbrick` — model interpretability
* :doc:`experiment` — experiment orchestration
* :doc:`utils` — utility functions
