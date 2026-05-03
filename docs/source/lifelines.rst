Lifelines Integration
=====================

Deckard provides specialized support for survival analysis through the optional
Lifelines integration. This enables time-to-event modeling, risk stratification,
and adversarial robustness studies on survival models.

.. _lifelines-overview:

Overview
--------

The Lifelines integration consists of four main modules:

- :mod:`deckard.data.survival` — survival dataset configuration with time/event pairs
- :mod:`deckard.model.survival` — lifelines estimator training and evaluation
- :mod:`deckard.score.survival` — survival-specific metrics (concordance, AIC, BIC)
- :mod:`deckard.experiment.survival` — end-to-end survival experiment orchestration

These modules support adversarial robustness studies on time-to-event models,
including attacks that perturb event times or event status.

Key Features
~~~~~~~~~~~~

- **Lifelines integration**: support for Kaplan-Meier, Cox PH, Weibull, and other
  lifelines estimators
- **Survival metrics**: concordance index, log-likelihood, AIC, BIC
- **Risk stratification**: stratified validation and group-wise metrics
- **Censoring support**: proper handling of censored observations
- **Time-aware attacks**: adversarial attacks on event times and event indicators
- **Survival curves**: plotting and visualization of survival functions
- **PyTorch integration**: optional deep learning survival models via extension

Survival Data
~~~~~~~~~~~~~

The :class:`~deckard.data.survival.LifelinesDataConfig` extends
:class:`deckard.data.DataConfig` with survival-specific fields:

- **duration_col** (str): column name for event times (durations)
- **event_col** (str): column name for event indicators (0 = censored, 1 = event)
- **auxiliary_model_config** (ModelConfig, optional): model to predict event
  times/status for inference attacks
- **stratify_by** (str, optional): column for stratified cross-validation

Survival Models
~~~~~~~~~~~~~~~

The :class:`~deckard.model.survival.SurvivalModelConfig` supports:

- Lifelines estimators: KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter, etc.
- Parametric and non-parametric models
- Risk stratification by covariates
- Partial hazard computation for attack generation
- Concordance index validation

Scoring and Metrics
~~~~~~~~~~~~~~~~~~~

The :mod:`deckard.score.survival` module provides:

- **concordance_index**: measure of prediction accuracy on time-to-event data
- **log_likelihood**: model fit quality
- **AIC**, **BIC**: model selection criteria
- **median_survival_time**: group-wise survival time estimates
- **survival_at_time_t**: proportion surviving at specific timepoints

Usage
-----

Command-line examples
~~~~~~~~~~~~~~~~~~~~~

**Basic survival experiment:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=survival \
      data.dataset_name=rossi \
      model=survival \
      model.model_type=lifelines.KaplanMeierFitter \
      score.model=survival

**Survival with Cox proportional hazards:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=survival \
      data.dataset_name=rossi \
      model=survival \
      model.model_type=lifelines.CoxPHFitter \
      score.model=survival

**Survival with attack evaluation:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=survival \
      data.dataset_name=rossi \
      model=survival \
      model.model_type=lifelines.CoxPHFitter \
      attack.attack_type=art.attacks.evasion.FastGradientMethod \
      attack.attack_params.eps=0.1 \
      score.model=survival

Programmatic examples
~~~~~~~~~~~~~~~~~~~~~

**Basic survival workflow:**

.. code-block:: python

   from deckard.data.survival import LifelinesDataConfig
   from deckard.model.survival import SurvivalModelConfig
   from deckard.experiment.survival import SurvivalExperimentConfig
   from deckard.score.survival import DefaultLifelinesConfig

   # Configure survival data
   data = LifelinesDataConfig(
       dataset_name="rossi",  # built-in lifelines dataset
       duration_col="week",
       event_col="arrest",
       train_size=70,
       test_size=30,
       random_state=42,
       scorer=DefaultLifelinesConfig(),
   )

   # Configure Cox proportional hazards model
   model = SurvivalModelConfig(
       model_type="lifelines.CoxPHFitter",
       duration_col="week",
       event_col="arrest",
       scorer=DefaultLifelinesConfig(),
   )

   # Run survival experiment
   cfg = SurvivalExperimentConfig(data=data, model=model)
   scores = cfg()

   print("Concordance Index:", scores.get("concordance_index"))
   print("AIC:", scores.get("aic"))
   print("Log-likelihood:", scores.get("log_likelihood"))

**Survival with Kaplan-Meier curves:**

.. code-block:: python

   from deckard.plot.survival import km_plot
   import matplotlib.pyplot as plt

   # Extract survival data and fitted model
   T = scores["durations"]
   E = scores["events"]
   survfunc = scores["survival_function"]

   # Plot Kaplan-Meier curve
   plt.figure(figsize=(10, 6))
   km_plot(T, E, ci_show=True)
   plt.title("Kaplan-Meier Survival Curve")
   plt.xlabel("Time (weeks)")
   plt.ylabel("Survival Probability")
   plt.tight_layout()
   plt.savefig("km_curve.png", dpi=300)
   plt.close()

**Survival with stratified validation:**

.. code-block:: python

   from deckard.data.survival import LifelinesDataConfig
   from deckard.model.survival import SurvivalModelConfig
   from deckard.experiment.survival import SurvivalExperimentConfig

   # Configure survival data with stratification
   data = LifelinesDataConfig(
       dataset_name="rossi",
       duration_col="week",
       event_col="arrest",
       stratify_by="prio",  # stratify by prior arrests
       train_size=70,
       test_size=30,
       scorer=DefaultLifelinesConfig(),
   )

   model = SurvivalModelConfig(
       model_type="lifelines.CoxPHFitter",
       duration_col="week",
       event_col="arrest",
       scorer=DefaultLifelinesConfig(),
   )

   cfg = SurvivalExperimentConfig(data=data, model=model)
   scores = cfg()

   # Access stratified concordance
   for group, group_scores in scores.get("group_metrics", {}).items():
       print(f"Group {group} - Concordance: {group_scores.get('concordance_index')}")

**Survival with time-aware attacks:**

.. code-block:: python

   from deckard.attack import AttackConfig
   from deckard.model.survival import SurvivalModelConfig

   # Configure attack that perturbs event times
   attack = AttackConfig(
       attack_type="art.attacks.evasion.FastGradientMethod",
       attack_params={"eps": 0.1},
       attack_size=50,
   )

   # Evaluate robustness of survival model
   cfg = SurvivalExperimentConfig(
       data=data,
       model=model,
       attack=attack,
   )
   scores = cfg()

   print("Original Concordance:", scores.get("concordance_index"))
   print("After Attack Concordance:", scores.get("attack_concordance", "N/A"))

Configuration
~~~~~~~~~~~~~

Key configuration options for :class:`~deckard.data.survival.LifelinesDataConfig`:

- **dataset_name** (str): name of lifelines dataset (rossi, load_rossi, etc.)
- **duration_col** (str): column name for event times
- **event_col** (str): column name for event indicators (0/1)
- **stratify_by** (str, optional): column for stratified sampling
- **auxiliary_model_config** (ModelConfig, optional): model for generating
  attack perturbations

For :class:`~deckard.model.survival.SurvivalModelConfig`:

- **model_type** (str): lifelines estimator (KaplanMeierFitter, CoxPHFitter,
  WeibullAFTFitter, etc.)
- **duration_col** (str): column name for durations
- **event_col** (str): column name for events
- **penalizer** (float, optional): regularization strength for Cox models
- **fit_method** (str, optional): optimization method (newton, bfgs, etc.)

Common Survival Metrics
~~~~~~~~~~~~~~~~~~~~~~~

- **concordance_index**: concordance between predicted and actual event ordering
  (1.0 = perfect, 0.5 = random)
- **log_likelihood**: log probability of observed data under model
- **AIC**: Akaike information criterion (lower is better)
- **BIC**: Bayesian information criterion (lower is better)
- **median_survival_time**: time at which 50% of population survived
- **event_rate**: proportion of observed events (vs. censored)
- **censoring_rate**: proportion of censored observations

Troubleshooting
~~~~~~~~~~~~~~~

- **Missing lifelines**: install via ``pip install lifelines`` or
  ``pip install "deckard[survival]"``
- **Duration/event column not found**: verify column names match actual dataset;
  check via ``data.X_train.columns``
- **Low concordance**: model may not fit data well; try different model types or
  feature engineering
- **Convergence warnings**: check for highly sparse or skewed durations; reduce
  penalizer or try different fit_method

See also
~~~~~~~~

* :doc:`data` — general data configuration including :mod:`deckard.data.survival`
* :doc:`model` — general model configuration including :mod:`deckard.model.survival`
* :doc:`score` — scoring framework including :mod:`deckard.score.survival`
* :doc:`experiment` — experiment orchestration including :class:`SurvivalExperimentConfig`
* :doc:`plot` — visualization including survival curve plotting
* :doc:`pytorch` — optional deep learning survival models
* :doc:`package` — overview of all extensions
