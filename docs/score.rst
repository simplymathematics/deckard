Score
=====

The :mod:`deckard.score` module defines scorer configuration objects used by
model, attack, and experiment pipelines.

.. automodule:: deckard.score
   :members:
   :show-inheritance:

Overview
--------

The score layer provides configurable scorer wrappers so data/model/attack
components can use a consistent scoring interface without hard-coding metric
implementations.

deckard now treats scoring as a runtime-configured layer rather than a fixed
set of metrics embedded inside each pipeline component.

- :class:`deckard.score.ScorerConfig` wraps a single metric callable.
- :class:`deckard.score.ScorerDictConfig` normalizes a mapping of metric names
   into callable scorer definitions.
- :class:`deckard.model.ModelConfig` and :class:`deckard.data.DataConfig` accept
   scorer configs directly through their ``scorer`` fields.
- :class:`deckard.attack.AttackConfig` delegates all attack scoring to
   :class:`deckard.score.attack.AttackScorerConfig`.

Attack scoring is now profile-based and attack-kind-aware:

- Evasion attacks use an evasion scorer profile and prefix outputs with
   ``evasion_``.
- Membership inference attacks use a membership scorer profile and prefix
   outputs with ``membership_inference_``.
- Attribute inference attacks use attribute scorer profiles and prefix outputs
   with ``inferred_<attribute>_``.
- Generic :class:`deckard.score.ScorerDictConfig` instances can be supplied to
   attack scorer profiles; deckard will route the correct ``y_true`` and
   ``y_pred`` values for the active attack kind and then prefix the resulting
   metric names.

The default score profiles registered in Hydra's config store are:

- ``scorers/classification``
- ``scorers/regression``
- ``scorers/fairness``
- ``scorers/survival``
- ``attack_scorers/evasion``
- ``attack_scorers/evasion-regression``
- ``attack_scorers/membership-inference``
- ``attack_scorers/attribute-inference``

These registrations are added through :func:`deckard.score.safe_store`, which
wraps Hydra's ``ConfigStore.instance().store(...)`` and tolerates duplicate
import-time registration attempts in tests and repeated imports.

Usage
-----

Programmatic example
~~~~~~~~~~~~~~~~~~~~

.. seealso::

   Fully-executed programmatic examples — including ``ScorerDictConfig``,
   model scoring, attack scoring, and fairness scoring — are available in
   the :doc:`notebooks/sklearn.ipynb </notebooks/sklearn>` notebook.

Model and data scoring
~~~~~~~~~~~~~~~~~~~~~~

:class:`~deckard.score.DefaultClassifierConfig` applies to model predictions.
Use :class:`~deckard.score.DefaultDataScoreConfig` (or leave ``scorer`` unset to
auto-select) when you want data-level statistics on the dataset itself.
See the :doc:`notebooks/sklearn.ipynb </notebooks/sklearn>` notebook for executed examples of both.

Score Modes and Routing
~~~~~~~~~~~~~~~~~~~~~~~

``ScorerDictConfig`` resolves ``y_true`` and ``y_pred`` from runtime context
using a mode-aware policy.

Supported modes include:

- ``train``: uses ``data.y_train`` and model training predictions.
- ``test``: uses ``data.y_test`` and model test predictions.
- ``val``: uses ``data.y_val`` and model validation predictions.
- ``attack`` / ``attack-val``: uses attack outputs and attack-aligned labels.
- ``pre-sample``: uses the full pre-split dataset (``data._X`` and
   ``data._y``) for dataset diagnostics.

For attack modes, label routing prefers ``attack.attacked_labels`` when
available, with split labels used as fallback when needed.

``pre-sample`` is intended for dataset diagnostics only. Metrics requiring
probability inputs (``needs_proba=True``) are rejected in this mode.

Attack scoring
~~~~~~~~~~~~~~

.. seealso::

   See the :doc:`notebooks/art_attacks.ipynb </notebooks/art_attacks>` notebook for a fully-executed attack
   scoring example with rendered score output.

In this example the evasion scorer profile reuses the generic classifier scorer
set. deckard passes the attack-specific prediction values to that generic
profile and prefixes the output keys so they remain unambiguous in merged score
dicts.

Multi-attack score key behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``ExperimentConfig.attack`` is configured as a list, deckard merges the
per-attack score dictionaries in order.

- Non-colliding keys are preserved as-is.
- Only colliding keys are suffixed with ``_<attack.alias>``.
- Aliases are required for multi-attack runs.

Example:

- ``evasion_accuracy`` from the first attack remains ``evasion_accuracy``
- a colliding ``evasion_accuracy`` from alias ``hsj`` becomes
   ``evasion_accuracy_hsj``

This naming behavior keeps backward compatibility for single-attack workflows
while preserving all metrics in multi-attack experiments.

Fairness Scoring Examples
~~~~~~~~~~~~~~~~~~~~~~~~~

Fairness score profiles are available in
`examples/sklearn/config/score/fairness-classification.yaml <../examples/sklearn/config/score/fairness-classification.yaml>`_ and
`examples/sklearn/config/score/fairness-regression.yaml <../examples/sklearn/config/score/fairness-regression.yaml>`_.

Classification fairness command:

.. code-block:: bash

   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name fairness-default \
      score=fairness-classification

Regression fairness command:

.. code-block:: bash

   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name fairness-default \
      score=fairness-regression

These profiles include metrics such as:

- ``demographic_parity_difference``
- ``equalized_odds_difference``
- ``group_mean_prediction_difference``
- ``group_mae_difference``
- ``group_mse_difference``

Attack + Fairlearn MetricFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For attack evaluations across sensitive groups, deckard supports
``FairlearnAttackScorerConfig``. This computes group-level attack metrics
through fairlearn's ``MetricFrame`` for evasion, membership inference, and
attribute inference outputs.

Programmatic example:

See the :doc:`notebooks/fairlearn.ipynb </notebooks/fairlearn>` notebook for an executed example using
``FairlearnAttackScorerConfig``.

Anjana Scoring Examples
~~~~~~~~~~~~~~~~~~~~~~~

Anjana scoring functions are available under :mod:`deckard.score.anjana` and
can be attached through :class:`deckard.score.ScorerDictConfig`.

Example scorer declaration:

.. code-block:: yaml

   scorers:
      k_anonymity:
         score_function: deckard.score.anjana.anjana_k_anonymity_score
      l_diversity:
         score_function: deckard.score.anjana.anjana_l_diversity_score
      t_closeness:
         score_function: deckard.score.anjana.anjana_t_closeness_score

This is useful when evaluating anonymization quality jointly with predictive
metrics and fairness/attack metrics in the same experiment run.

Hydra ConfigStore examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can select canonical scorer profiles directly from Hydra's config store.
The integration tests validate these score groups with overrides such as
``score=classification``, ``score=regression``, ``score=survival``,
``score=fairness-classification``, and ``score=fairness-regression``.

Example experiment config:

.. code-block:: yaml

   defaults:
      - _self_
      - score: classification

   score:
      scorers:
         accuracy:
            score_function: sklearn.metrics.accuracy_score
         precision:
            score_function: sklearn.metrics.precision_score
            score_params:
               average: weighted
               zero_division: 0

Example CLI overrides:

.. code-block:: bash

   python -m deckard optimize --config-name experiment score=classification
   python -m deckard optimize --config-name experiment score=regression
   python -m deckard optimize --config-name survival score=survival
   python -m deckard optimize --config-name experiment data=fair-adult score=fairness-classification
   python -m deckard optimize --config-name experiment data=fair-adult score=fairness-regression

Examples with attack scorer profiles:

.. code-block:: bash

   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name attack-default \
      score=attack-classification \
      attack=fgm

   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name attribute-inference-default \
      score=attribute-inference

   python -m deckard optimize \
      --config-path examples/sklearn/config \
      --config-name inference-default \
      score=membership-inference

You can also override nested scorer definitions inline when you only need one
or two metrics:

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      'model.scorer.scorers.accuracy.score_function=sklearn.metrics.accuracy_score' \
      'attack.scorer.evasion.scorers.success.score_function=deckard.score.attack.evasion_success_score'

For attribute inference against continuous targets there is currently no
separate canonical Hydra store alias for
``attribute_inference_regression``. Configure that profile inline on
``attack.scorer.attribute_inference_regression`` when needed.

Direct ScorerDictConfig Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also instantiate :class:`deckard.score.ScorerDictConfig` directly
for fine-grained metric customization. See the :doc:`notebooks/sklearn.ipynb </notebooks/sklearn>`
notebook for a fully-executed example with rendered metric output.

.. code-block:: python

   results = {}
   for scorer_name, scorer_fn in callables.items():
      try:
         results[scorer_name] = scorer_fn(y_true=y_true, y_pred=y_pred)
      except Exception as e:
         print(f"Scorer {scorer_name} failed: {e}")

   print(f"Results: {results}")

Internals
---------

Score configs normalize definitions into callable maps and support both
classification and regression defaults through dedicated config classes.

The main scoring flow is:

1. A config object normalizes metric declarations into
    :class:`deckard.score.ScorerConfig` instances.
2. :class:`deckard.score.ScorerDictConfig` resolves import-string callables,
    filters unsupported keyword arguments against the target metric signature,
    and executes the metric.
3. Pipeline components decide which targets and predictions to pass.
4. Attack scoring adds attack-kind-specific prefixes and timing fields.

Important attack-scoring details:

- :class:`deckard.score.attack.AttackScorerConfig` owns all attack scoring
   behavior.
- ``score_evasion`` chooses between classification and regression evasion
   profiles based on the detected task type.
- ``score_membership`` evaluates inferred membership labels against the attack
   labels (``attack.attacked_labels``).
- ``score_attribute`` chooses categorical vs regression attribute profiles and
   prefixes metrics with the targeted attribute name.
- All attack score dicts add ``attack_size`` and ``attack_score_time``; some
   attribute paths also include ``attack_generation_time``.

Because scorer profiles are normal config objects, the same metric definition
can be reused across model scoring and attack scoring. The attack layer adapts
the inputs and prefixes the outputs instead of requiring a separate metric
implementation for every attack family.

Troubleshooting
---------------

- Ensure metric names map to importable scorer callables.
- Check expected prediction shape/type for selected metrics.
- Confirm task type (classifier/regressor) matches the active default scorer set.
- If a metric needs extra keyword arguments, ensure the callable accepts them;
   deckard drops unsupported keyword arguments based on the metric signature.
- If Hydra overrides appear to be ignored, verify that you are targeting the
   correct store group: ``scorers/...`` for model/data/fairness/survival and
   ``attack_scorers/...`` for attack profiles.
- If attack metrics are missing, look for prefixed keys such as ``evasion_*``,
   ``membership_inference_*``, or ``inferred_age_*`` rather than the raw metric
   names.
- If evasion scoring is configured with regression metrics for a classification
   attack, or vice versa, the resulting metric calculations may fail or produce
   misleading values. Match the scorer profile to the task type.


See also
~~~~~~~~

* :doc:`model` — model configuration and evaluation
* :doc:`data` — data configuration
* :doc:`attack` — attack scoring
* :doc:`experiment` — experiment orchestration
* :doc:`lifelines` — survival-specific metrics
* :doc:`anjana` — anonymization metrics
