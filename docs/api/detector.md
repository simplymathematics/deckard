# Detector

## Introduction

This page is the canonical home for detector module behavior and API details.
It covers detector lifecycle stages, filter side effects, persistence paths,
and integration with attack/model scoring.

## Overview

The detector module defines detector-specific configuration objects used to
evaluate detector behavior against experiment outputs.

It is typically used after model and attack execution to derive detector-level
metrics from benign and attacked samples.

Canonical runtime contract:

- files: detector artifacts persist through files-only paths (`detector_model_file`, `detected_predictions_file`, `score_file`)
- times: `detector_training_time`, `detector_detection_time`
- scores: detector-prefixed metrics merged with runtime timing metadata
- stage: canonical detector stage tokens (`pre-fit`, `post-fit`, `pre-detect`, `post-detect`)
- ordering: detector execution is marked as `post-attack` for consistent orchestration metadata

## Examples

```{seealso}

   Notebook-based detector workflows are documented in:

   - {doc}`notebooks/detector.ipynb </notebooks/detector>`
   - {doc}`notebooks/art_attacks.ipynb </notebooks/art_attacks>`

```

## API Reference

```{eval-rst}
.. automodule:: deckard.detector
   :members:
   :show-inheritance:
```

## Minimal YAML Example

```yaml
detector:
   _target_: deckard.detector.base.DetectorConfig
   detector_type: art.defences.detector.evasion.BinaryInputDetector
```

## Typical Workflow

1. Prepare model/data outputs via the experiment layer.
1. Optionally generate attack artifacts.
1. Run detector scoring on benign and attacked outputs.

## Troubleshooting

- Confirm detector configuration is compatible with the selected task/backend.
- Ensure upstream experiment outputs are present before detector execution.
- Verify detector score keys do not collide with model/attack score keys.

### See also

- {doc}`experiment` — experiment orchestration
- {doc}`attack` — attack generation and attack outputs
- {doc}`score` — scoring framework including detector metrics
