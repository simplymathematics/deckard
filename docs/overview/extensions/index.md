# Extensions

This page maps Deckard's optional extension ecosystem. Use it after the core
overview pages when you need framework-specific execution behavior or
plugin-specific trustworthiness workflows.

For full hook ownership and execution-policy details, see
[Plugin and Hook Execution Reference](../../developers/hooks).

## Frameworks

Framework integrations extend the base data, model, attack, and scoring
contracts with runtime-specific execution details.

### sklearn

The default tabular workflow composes:

- [Data API](../../api/data) for sampling and preprocessing pipelines
- [Model API](../../api/model) for trainer selection and persistence
- [Attack API](../../api/attack) for robustness evaluation
- [Score API](../../api/score) for metric composition

Use [sklearn notebook](../../notebooks/sklearn) for an end-to-end example.
See also [sklearn framework overview](sklearn).

### PyTorch

Torch-native workflows add dataloaders, tensor models, and trainer-specific
execution on top of the same core contracts.

- [PyTorch API](../../api/pytorch)
- [Attack API](../../api/attack)
- [Score API](../../api/score)

Use [pytorch notebook](../../notebooks/pytorch) for runnable examples.
See also [PyTorch framework overview](pytorch).

### Transformers

Transformer workflows build on tokenization, encoded feature payloads, and
transformer-specific trainer/runtime adapters.

- [Pipeline API](../../api/pipeline)
- [Model API](../../api/model)
- [Attack API](../../api/attack)
- [Score API](../../api/score)

See also [Transformers framework overview](transformers).

## Plugins

Plugin integrations specialize the base runtime for fairness, privacy,
survival analysis, and visualization.

### Anjana

Anjana adds anonymization-aware preprocessing and privacy-oriented scoring.

- [Anjana API](../../api/anjana)
- [Anjana notebook](../../notebooks/anjana)

See also [ANJANA plugin overview](anjana).

### Fairlearn

Fairlearn adds sensitive-feature-aware data handling and group fairness
metrics.

- [Fairlearn API](../../api/fairlearn)
- [Fairlearn notebook](../../notebooks/fairlearn)

See also [Fairlearn plugin overview](fairlearn).

### Lifelines

Lifelines adds survival analysis and time-to-event model workflows.

- [Lifelines API](../../api/lifelines)
- [Lifelines notebook](../../notebooks/lifelines)

See also [Lifelines plugin overview](lifelines).

### Seaborn and Yellowbrick

Seaborn and Yellowbrick add reporting, diagnostics, and visualization layers on
top of persisted experiment outputs.

- [Seaborn API](../../api/seaborn)
- [Yellowbrick API](../../api/yellowbrick)
- [Seaborn notebook](../../notebooks/seaborn)
- [Yellowbrick notebook](../../notebooks/yellowbrick)

See also [Seaborn plugin overview](seaborn) and
[Yellowbrick plugin overview](yellowbrick).

### ART-backed robustness workflows

Robustness workflows rely on the core attack and defense surfaces plus notebook
examples rather than a standalone plugin page.

- [Attack API](../../api/attack)
- [Defense API](../../api/defend)
- [art_attacks notebook](../../notebooks/art_attacks)
- [art_defenses notebook](../../notebooks/art_defenses)
- [detector notebook](../../notebooks/detector)

```{toctree}
:hidden:

sklearn
pytorch
transformers
anjana
fairlearn
lifelines
seaborn
yellowbrick
```
