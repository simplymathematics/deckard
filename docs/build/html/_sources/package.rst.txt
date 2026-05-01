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

The package-level API exposes the core configuration classes that drive this
workflow:

- :class:`deckard.data.DataConfig`
- :class:`deckard.model.ModelConfig` and :class:`deckard.model.DefenseConfig`
- :class:`deckard.attack.AttackConfig`
- :class:`deckard.experiment.ExperimentConfig`
- :class:`deckard.file.FileConfig`
- :class:`deckard.score.ScorerDictConfig`

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

This reduces engineering friction so researchers can focus on methodology
instead of ad-hoc pipeline glue code.

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

The [examples](../../examples) directory contains reproducible experiment
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

1. dataset loading/sampling
2. preprocessing and feature handling
3. model training/evaluation
4. optional defenses
5. attack execution
6. scoring and artifact persistence

By standardizing these stages, Deckard reduces ambiguity in experiment setup
and makes comparative benchmarking easier.

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

This enables scalable robustness studies without rewriting experiment code for
each execution backend.

The same design also supports non-robustness-focused studies by swapping in
new metrics, transforms, defenses, and task-specific evaluation components.

Internals
---------

At import time, the package registers OmegaConf resolvers (for example
``file``, ``merge``, and ``hash``), installs warning filters, and exports the
public API listed in ``deckard.__all__``.

Troubleshooting
---------------

- If interpolations fail, verify resolver inputs and config paths.
- If optional backends are unavailable, install the corresponding extras.
- If behavior differs between runs, compare resolved configs and hash values.

See also
~~~~~~~~

* :doc:`experiment`
* :doc:`modules`
* :doc:`utils`
