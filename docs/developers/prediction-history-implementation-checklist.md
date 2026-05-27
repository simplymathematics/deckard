# Prediction History Implementation Checklist

## Objective
- [ ] Preserve per-fold and per-split scoring correctness.
- [ ] Retain predictions and probabilities across all repeated runs.
- [ ] Keep backward compatibility for existing score outputs and tests.

## Phase 1: Data Model and Runtime State
- [ ] Add an experiment-level runtime container for prediction history.
- [ ] Define canonical history shape keyed by mode and run index.
- [ ] Include per-run payload fields:
  - [ ] y_true
  - [ ] y_pred
  - [ ] y_proba (optional)
  - [ ] split or fold metadata (run index, sampler type, optional indices)
- [ ] Ensure active prediction fields remain single-run only and are not extended across runs.

## Phase 2: Capture Per-Run Outputs
- [ ] Add helper to capture outputs for the current run after model scoring.
- [ ] Capture outputs for each active mode used by experiment scoring.
- [ ] Store payload under history[mode][run_idx].
- [ ] Add guardrails for missing y_proba when model does not support predict_proba.

## Phase 3: Keep Per-Fold Scoring Pure
- [ ] Verify scoring always consumes current run payloads only.
- [ ] Prevent scorers from reading concatenated history payloads during fold-level scoring.
- [ ] Confirm fold or split score keys continue to represent one run only.

## Phase 4: Aggregate History-Derived Metrics (Optional but Recommended)
- [ ] Add helper to build merged payloads from prediction history for aggregate reporting.
- [ ] Compute aggregate metrics in a separate namespace (for example overall or oof).
- [ ] Do not overwrite fold-N or split-N keys.
- [ ] Support numeric mean aggregation and retain non-numeric values deterministically.

## Phase 5: Output Contract and Serialization
- [ ] Keep existing top-level fold or split score schema unchanged.
- [ ] Add new top-level key for history-derived outputs.
- [ ] Ensure history payload is serializable in current artifact pipeline.
- [ ] Validate behavior when caching is enabled and disabled.

## Phase 6: Tests
- [ ] Add unit test for history container initialization and shape.
- [ ] Add test that repeated runs create one history entry per run index.
- [ ] Add test that fold-level scores are unchanged relative to baseline behavior.
- [ ] Add test that aggregate history metrics appear under separate keys.
- [ ] Add test for classifier without probabilities to verify graceful handling.
- [ ] Add regression test for KFold repeated runs.
- [ ] Add regression test for Shuffle repeated runs.

## Phase 7: Documentation
- [ ] Document difference between active run fields and prediction history fields.
- [ ] Document new aggregate namespace and intended interpretation.
- [ ] Add example output snippet showing fold keys and overall keys.
- [ ] Update developer docs with migration notes for downstream consumers.

## Phase 8: Validation and Rollout
- [ ] Run focused experiment tests for repeated run classes.
- [ ] Run full experiment test module.
- [ ] Run targeted docs build if any docs were changed.
- [ ] Validate no new lints or type errors in touched files.
- [ ] Prepare concise changelog entry summarizing behavior and compatibility.

## Definition of Done
- [ ] All repeated-run tests pass.
- [ ] Per-fold and per-split metrics remain correct and isolated.
- [ ] Historical predictions are available for all runs.
- [ ] Aggregate history metrics are exposed under separate keys.
- [ ] Existing consumers of current score keys are not broken.
