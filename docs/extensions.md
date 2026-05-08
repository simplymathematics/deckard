# Extensions

This page documents the main extension modules and optional backends in deckard.

## Data Extensions
- Fairlearn integration: `deckard.data.fairness`
- Anjana anonymization: `deckard.data.anjana`
- PyTorch integration: `deckard.data.pytorch`
- Survival analysis: `deckard.data.survival`

## Model Extensions
- Fairlearn fairness models: `deckard.model.fairness`
- Anjana anonymization models: `deckard.model.anjana`
- PyTorch models: `deckard.model.pytorch`
- Survival models: `deckard.model.survival`
- ART defense pipeline: `deckard.model.defend`

## Scoring Extensions
- Fairness metrics: `deckard.score.fairness`
- Anonymization metrics: `deckard.score.anjana`
- Survival metrics: `deckard.score.survival`
- Attack metrics: `deckard.score.attack`
- Data metrics: `deckard.score.data`

## Other Extensions
- Attacks: `deckard.attack`
- Visualization: `deckard.plot` (seaborn, yellowbrick, survival curves)
- Advanced workflows: `deckard.layers`
- PyTorch experiment orchestration: `deckard.experiment.torch_experiment`
- Survival experiment orchestration: `deckard.experiment.survival`
