Detector
========

The :mod:`deckard.detector` module provides auxiliary detector orchestration for
post-attack analysis. Detector models are trained to classify **clean vs
adversarial/poisoned** samples and are executed as an optional phase in
:class:`deckard.experiment.ExperimentConfig`.

.. automodule:: deckard.detector
   :members:
   :show-inheritance:

Overview
--------

Detector execution is intentionally separate from model defenses:

- **Defense pipeline**: transforms or wraps the task model used for the
  original prediction task.
- **Detector phase**: trains a secondary model for detection after adversarial
  samples are generated.

In ExperimentConfig, detector execution runs after attack execution and merges
metrics under ``detector_*`` keys.

Supported detector families
---------------------------

- Evasion detectors exposing ``fit`` + ``detect`` APIs
  (for example ``BinaryInputDetector``).
- Poison detectors exposing ``detect_poison`` APIs
  (for example ``SpectralSignatureDefense``).

Configuration examples
----------------------

Binary input detector (evasion):

.. code-block:: yaml

   detector_type: art.defences.detector.evasion.BinaryInputDetector
   detector_params: {}
   fit_params:
     batch_size: 16
     nb_epochs: 1
     split: test
   detector_model:
     model_type: sklearn.linear_model.LogisticRegression
     classifier: true
     model_params:
       max_iter: 50
   _target_: deckard.detector.DetectorConfig

Spectral signature detector (poison):

.. code-block:: yaml

   detector_type: art.defences.detector.poison.SpectralSignatureDefense
   detector_params:
     expected_pp_poison: 0.2
     batch_size: 16
   fit_params:
     split: train
   detector_model:
     model_type: sklearn.linear_model.LogisticRegression
     classifier: true
     model_params:
       max_iter: 100
   _target_: deckard.detector.DetectorConfig
