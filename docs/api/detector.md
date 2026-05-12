# Detector

## Overview

The detector module defines detector-specific configuration objects used to
evaluate detector behavior against experiment outputs.

It is typically used after model and attack execution to derive detector-level
metrics from benign and attacked samples.

## Examples

.. seealso::

   Notebook-based detector workflows are documented in:

   - :doc:`notebooks/detector.ipynb </notebooks/detector>`
   - :doc:`notebooks/art_attacks.ipynb </notebooks/art_attacks>`

## API Reference

.. automodule:: deckard.detector
   :members:
   :show-inheritance:

.. automodule:: deckard.detector.base
   :members:
   :show-inheritance:

## Typical Workflow

1. Prepare model/data outputs via the experiment layer.
2. Optionally generate attack artifacts.
3. Run detector scoring on benign and attacked outputs.

## Troubleshooting

- Confirm detector configuration is compatible with the selected task/backend.
- Ensure upstream experiment outputs are present before detector execution.
- Verify detector score keys do not collide with model/attack score keys.

### See also

* :doc:`experiment` — experiment orchestration
* :doc:`attack` — attack generation and attack outputs
* :doc:`score` — scoring framework including detector metrics
