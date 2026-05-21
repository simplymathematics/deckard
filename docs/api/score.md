# Score

The {mod}`deckard.score` module defines scorer configuration objects used by
model, attack, and experiment pipelines.

```{eval-rst}
.. automodule:: deckard.score
   :members:
   :show-inheritance:
```


## Overview

The score layer provides configurable scorer wrappers so data/model/attack
components can use a consistent scoring interface without hard-coding metric
implementations.

deckard now treats scoring as a runtime-configured layer rather than a fixed
set of metrics embedded inside each pipeline component.

- {class}`~deckard.score.ScorerConfig` wraps a single metric callable.
- {class}`~deckard.score.ScorerDictConfig` normalizes a mapping of metric names
   into callable scorer definitions.
- {class}`~deckard.model.ModelConfig` and {class}`~deckard.data.DataConfig` accept
   scorer configs directly through their ``scorer`` fields.
- {class}`~deckard.attack.AttackConfig` delegates all attack scoring to
   {class}`~deckard.score.attack.AttackScorerConfig`.

Attack scoring is now profile-based and attack-kind-aware:

- Evasion attacks use an evasion scorer profile and prefix outputs with
   ``evasion_``.
- Membership inference attacks use a membership scorer profile and prefix
   outputs with ``membership_inference_``.
- Attribute inference attacks use attribute scorer profiles and prefix outputs
   with ``inferred_<attribute>_``.
- Generic {class}`~deckard.score.ScorerDictConfig` instances can be supplied to
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

These registrations are added through :func:`~deckard.score.safe_store`, which
wraps Hydra's ``ConfigStore.instance().store(...)`` and tolerates duplicate
import-time registration attempts in tests and repeated imports.

## Common scorer components and references

Specific scorer profiles in this module are designed to compose with
{doc}`model`, {doc}`data`, {doc}`attack`, and plugin APIs such as
{doc}`fairlearn` and {doc}`lifelines`.

Frequently referenced metric callables include:

- [`sklearn.metrics.accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [`sklearn.metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [`sklearn.metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
- [`fairlearn.metrics.demographic_parity_difference`](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.demographic_parity_difference.html)
- [`fairlearn.metrics.equalized_odds_difference`](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.equalized_odds_difference.html)
- [`lifelines.utils.concordance_index`](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index)

## Examples

```{seealso}

   Notebook-based scoring workflows (model metrics, data diagnostics,
   attack scoring, and fairness scoring) are documented in:

   - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - {doc}`notebooks/art_attacks.ipynb </notebooks/art_attacks>`
   - {doc}`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`

```
## Internals

Score configs normalize definitions into callable maps and support both
classification and regression defaults through dedicated config classes.

The main scoring flow is:

1. A config object normalizes metric declarations into
   {class}`~deckard.score.ScorerConfig` instances.
2. {class}`~deckard.score.ScorerDictConfig` resolves import-string callables,
    filters unsupported keyword arguments against the target metric signature,
    and executes the metric.
3. Pipeline components decide which targets and predictions to pass.
4. Attack scoring adds attack-kind-specific prefixes and timing fields.

Important attack-scoring details:

- {class}`~deckard.score.attack.AttackScorerConfig` owns all attack scoring
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

## Troubleshooting

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


### See also

* {doc}`model` — model configuration and evaluation
* {doc}`data` — data configuration
* {doc}`attack` — attack scoring
* {doc}`experiment` — experiment orchestration
* {doc}`lifelines` — survival-specific metrics
* {doc}`anjana` — anonymization metrics
