# Model Runtime Canon

Deckard's model runtime is intentionally split into a small set of canonical responsibilities:

- resolve and train or load a model
- apply defenses at the correct stage
- preserve timing, predictions, and score state across reruns
- keep framework- and plugin-specific behavior in wrappers, not in the core runtime

This page documents the model-side runtime contract that the refactor enforces.

## Canonical Model Shape

A model config should expose a stable runtime surface whether it is the core sklearn-style `ModelConfig` or a framework/plugin specialization.

Canonical runtime fields include:

- `_model`
- `score_dict`
- `training_time`
- `prediction_time`
- `training_predictions`
- `predictions`
- `training_probabilities`
- `probabilities`
- `defense_application_time`
- `training_n`
- `prediction_n`

The runtime is expected to be able to load, train, score, and persist without requiring callers to understand the underlying framework.

## Canonical Public Methods

Framework and plugin wrappers must expose the same public runtime verbs as base configs.

- Model runtime: `train`, `predict`, `predict_proba`, `score`, `apply_defense`
- Detector runtime: `filter` for filter-mode execution
- Defense runtime mixins: verb-form handlers such as `defend`, `detect`, `preprocess`, `postprocess`, `regularize`, `train_defense`

Underscored method variants are considered internal implementation details and
must not be used as the public orchestration surface in tests or docs.

## Trainer Canon

Training is owned by a trainer runtime object instead of by scattered branching in the config classes.

The trainer contract is intentionally simple:

- resolve a trainer from configuration
- compose the resolved runtime object
- execute training with the data runtime and optional persistence state

This is what makes pretrained-model reruns and forced retraining possible without duplicating the orchestration logic.

### Pretrained reruns

When a pretrained model receives a defense that must run at fit time (`apply_fit=True`), Deckard now:

- snapshots the pre-defense runtime state
- clears cached predictions
- retrains before applying the fit-time defense
- stores the pre-defense metrics under an explicit key such as `pre-defense` or `pre-<alias>-defense`

That keeps the old state available for analysis while ensuring the defense is applied against a freshly trained estimator.

## Defense Canon

Model defenses are applied by stage, not by ad hoc backend checks.

Canonical model defense stages:

- `pre_art_defense`
- `pre_fit`
- `post_fit_pre_predict`

Stage selection is driven by the defense family:

- ANJANA data defenses are treated as pre-ART model-stage behavior
- `fairlearn.reductions` defenses run in the pre-fit stage
- `fairlearn.adversarial` defenses run after fit, before predict

Deckard also preserves the fit-time versus predict-time distinction for defense wrappers:

- fit-time defenses can trigger retraining
- predict-time defenses are applied after model loading/training
- wrapper-stage reapplication must not duplicate existing defenses

## Persistence and State

Model persistence follows the same broad pattern as data persistence:

- config objects remain YAML-serializable
- runtime artifacts stay framework-specific
- canonical timing and score keys are preserved across reruns
- score and timing snapshots are kept before defense-triggered retraining

The important rule is that persistence should not leak framework-specific orchestration details into the top-level API.

## Framework and Plugin Boundaries

Framework and plugin model configs should remain thin wrappers around the canonical model runtime.

That means they may own:

- device handling
- framework-specific estimator construction
- sensitive-feature routing
- fairness or privacy policy hooks
- framework-native checkpoint formats

They should not reimplement the core train/load/defense orchestration.

## Related Docs

- {doc}`../api/model`
- {doc}`../api/pytorch`
- {doc}`../api/fairlearn`
- {doc}`../api/anjana`
- {doc}`../overview/model`
- {doc}`../developers/plugins`
- {doc}`../developers/data`
