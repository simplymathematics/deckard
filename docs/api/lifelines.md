# Lifelines Integration

deckard provides specialized support for survival analysis through the optional
Lifelines integration. This enables time-to-event modeling, risk stratification,
and adversarial robustness studies on survival models.

(lifelines-overview)=

## Overview

The Lifelines integration consists of four main modules:

- {mod}`deckard.plugins.lifelines.data` — survival dataset configuration with
  time/event pairs
- {mod}`deckard.plugins.lifelines.model` — lifelines estimator training and evaluation
- {mod}`deckard.plugins.lifelines.score` — survival-specific metrics
  (concordance, AIC, BIC)
- {mod}`deckard.plugins.lifelines.experiment` — end-to-end survival experiment orchestration

These modules support adversarial robustness studies on time-to-event models,
including attacks that perturb event times or event status.

Hydra-oriented configuration for survival experiments is covered in
{doc}`experiment` and follows [Hydra](https://hydra.cc) composition patterns.

External references:

- [lifelines documentation](https://lifelines.readthedocs.io)
- [`lifelines.fitters.coxph_fitter.CoxPHFitter`](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html)
- [`lifelines.fitters.weibull_fitter.WeibullFitter`](https://lifelines.readthedocs.io/en/latest/fitters/univariate/WeibullFitter.html)
- [`lifelines.utils.concordance_index`](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index)

Related Deckard docs:

- {doc}`score` for shared scorer profiles and metric composition
- {doc}`model` for defense-enabled model orchestration
- {doc}`attack` for attack execution with survival configurations
- {doc}`plot` for survival curve plotting backends
- {doc}`pipeline` and {doc}`sample` for survival preprocessing/split workflows

### Key Features

- **Lifelines integration**: support for Kaplan-Meier, Cox PH, Weibull, and other
  lifelines estimators
- **Survival metrics**: concordance index, log-likelihood, AIC, BIC
- **Risk stratification**: stratified validation and group-wise metrics
- **Censoring support**: proper handling of censored observations
- **Time-aware attacks**: adversarial attacks on event times and event indicators
- **Survival curves**: plotting and visualization of survival functions
- **PyTorch integration**: optional deep learning survival models via extension

Extension docs:

- {doc}`pytorch`
- {doc}`plot`
- {doc}`experiment`

### Score Types Available

The default survival scorer profile is
{class}`deckard.plugins.lifelines.score.DefaultLifelinesConfig` and includes:

- `concordance` via :func:`~deckard.plugins.lifelines.score.survival_concordance_score`
- `aic` via :func:`~deckard.plugins.lifelines.score.survival_aic_score`
- `bic` via :func:`~deckard.plugins.lifelines.score.survival_bic_score`

These are also provided in the sklearn example score profile at
`examples/sklearn/config/score/survival.yaml <../examples/sklearn/config/score/survival.yaml>`\_.

### Survival Data

The {class}`~deckard.plugins.lifelines.data.LifelinesDataConfig` extends
{class}`deckard.data.DataConfig` with survival-specific fields:

- **duration_col** (str): column name for event times (durations)
- **event_col** (str): column name for event indicators (0 = censored, 1 = event)
- **auxiliary_model_config** (ModelConfig, optional): model to predict event
  times/status for inference attacks
- **stratify_by** (str, optional): column for stratified cross-validation

Survival data mode support is explicit in
{class}`deckard.plugins.lifelines.data.LifelinesDataConfig`:

- `native`: dataset already has duration/event columns
- `auxiliary_model`: derive failure events from a benign model metric
- `auxiliary_attack`: derive failures from attack outputs
- `optuna_db`: treat Optuna study outputs as time-to-event data

### Data pipeline and sampling support

Because {class}`~deckard.plugins.lifelines.data.LifelinesDataConfig` extends
{class}`~deckard.data.DataConfig` (through the deckard data
stack), survival workflows can still use the standard data pipeline and sampler
interfaces:

- preprocessing pipelines from {class}`~deckard.data.DataPipelineConfig`
- split/k-fold/shuffle samplers via `examples/sklearn/config/sample <../examples/sklearn/config/sample>`\_
- train/test/validation flow from core data config fields

This lets users mix survival-specific fields (duration/event/mode) with normal
deckard preprocessing and split strategies.

### Survival Models

The {class}`~deckard.plugins.lifelines.model.SurvivalModelConfig` supports:

- Lifelines estimators: KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter, etc.
- Parametric and non-parametric models
- Risk stratification by covariates
- Partial hazard computation for attack generation
- Concordance index validation

### Scoring and Metrics

The {mod}`deckard.plugins.lifelines.score` module provides:

- **concordance_index**: measure of prediction accuracy on time-to-event data
- **log_likelihood**: model fit quality
- **AIC**, **BIC**: model selection criteria
- **median_survival_time**: group-wise survival time estimates
- **survival_at_time_t**: proportion surviving at specific timepoints

When using {class}`deckard.plugins.lifelines.model.SurvivalModelConfig` without
a custom
scorer override, model scoring still emits calibration-oriented metrics (for
example `concordance`, `ici`, `e50`) where available.

### Defenses in survival workflows

Survival models can use deckard's defense pipeline from
{class}`deckard.model.defend.DefensePipelineConfig` just like other model types.
Supported defense families include ART preprocessors, postprocessors,
detectors, and trainers.

Typical usage pattern:

- choose a survival model (for example `lifelines.fitters.coxph_fitter.CoxPHFitter`)
- attach `model.defense` entries from `examples/sklearn/config/defense <../examples/sklearn/config/defense>`\_
- evaluate robustness with survival scores and optional attacks in the same run

### Survival experiment contract

{class}`~deckard.plugins.lifelines.experiment.SurvivalExperimentConfig` requires
these
fields at construction time:

- `data`
- `model` (string model name/alias, for example `cox` or `weibull`)
- `target`
- `event_col`
- `duration_col`

In YAML configs, `model_type` should be a fully-qualified import path so
custom user-provided regression fitters can be imported reliably.

## Examples

```{seealso}

   Notebook-based survival workflows (lifelines estimators, survival scoring,
   and survival plotting) are documented in:

  - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`

```

### Troubleshooting

- **Missing lifelines**: install via `pip install lifelines` or
  `pip install "deckard[survival]"`
- **Duration/event column not found**: verify column names match actual dataset;
  check via `data.X_train.columns`
- **Low concordance**: model may not fit data well; try different model types or
  feature engineering
- **Convergence warnings**: check for highly sparse or skewed durations; reduce
  penalizer or try different fit_method

### See also

- {doc}`data` — general data configuration including {mod}`deckard.plugins.lifelines.data`
- {doc}`model` — general model configuration including {mod}`deckard.plugins.lifelines.model`
- {doc}`score` — scoring framework including {mod}`deckard.plugins.lifelines.score`
- {doc}`experiment` — experiment orchestration including
  {class}`~deckard.plugins.lifelines.experiment.SurvivalExperimentConfig`
- {doc}`plot` — visualization including survival curve plotting
- {doc}`pipeline` — data-pipeline composition for survival datasets
- {doc}`sample` — split and fold strategies used by survival runs
- {doc}`pytorch` — optional deep learning survival models
- {doc}`modules` — overview of all extensions
