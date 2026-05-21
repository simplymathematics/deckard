# Extensions

This page maps Deckard extension points to both internal API docs and external
library references.

## Core Pipelines

### sklearn transform pipelines

Deckard data preprocessing pipelines are configured through
[Data API](../api/data) and model orchestration in [Model API](../api/model).
These map directly to sklearn pipeline concepts:

- [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [`sklearn.compose.ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

For end-to-end examples, see [sklearn notebook](../notebooks/sklearn).

### torch transform pipelines

Torch-native data/model pipelines are documented in [PyTorch API docs](../api/pytorch),
with attack/scoring integration through [Attack API](../api/attack) and
[Score API](../api/score).

External references:

- [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [`torchvision.transforms.Compose`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html)
- [`torch.nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)

For runnable examples, see [pytorch notebook](../notebooks/pytorch).

## ART Attacks and Defenses

Deckard attack orchestration is documented in [Attack API docs](../api/attack),
and defense orchestration in [Model API docs](../api/model).

Common ART attack references used in Deckard workflows:

- [`FastGradientMethod`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm)
- [`ProjectedGradientDescent`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#projected-gradient-descent-pgd)
- [`BasicIterativeMethod`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#basic-iterative-method-bim)

Common ART defense references used in Deckard workflows:

- [`FeatureSqueezing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#feature-squeezing)
- [`SpatialSmoothing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#spatial-smoothing)
- [`AdversarialTrainer`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/trainer.html#adversarial-training)

Notebook walkthroughs:

- [art_attacks notebook](../notebooks/art_attacks)
- [art_defenses notebook](../notebooks/art_defenses)

## Scorers and Metrics

Score configuration and attack-aware scorer profiles are documented in
[Score API docs](../api/score) and compose with [Attack API](../api/attack),
[Model API](../api/model), and [Data API](../api/data).

External scorer references:

- [`sklearn.metrics.accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [`sklearn.metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [`sklearn.metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
- [`fairlearn.metrics.demographic_parity_difference`](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.demographic_parity_difference.html)
- [`lifelines.utils.concordance_index`](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index)

## Plugins

### [Anjana](../api/anjana)

Integration for anonymization-aware machine learning.

- Internal docs: [Anjana API docs](../api/anjana)
- External docs: [Anjana on PyPI](https://pypi.org/project/anjana/)

### [Fairlearn](../api/fairlearn)

Integration for fairness-aware machine learning.

- Internal docs: [Fairlearn API docs](../api/fairlearn)
- External docs: [Fairlearn documentation](https://fairlearn.org/main/)

### [Lifelines](../api/lifelines)

Integration for survival analysis and time-to-event modeling.

- Internal docs: [Lifelines API docs](../api/lifelines)
- External docs: [lifelines documentation](https://lifelines.readthedocs.io)

### [Seaborn](../api/seaborn)

Statistical visualization with Seaborn.

- Internal docs: [Seaborn API docs](../api/seaborn)
- External docs: [Seaborn documentation](https://seaborn.pydata.org)

### [Yellowbrick](../api/yellowbrick)

Single-run model diagnostics with Yellowbrick.

- Internal docs: [Yellowbrick API docs](../api/yellowbrick)
- External docs: [Yellowbrick documentation](https://www.scikit-yb.org)

## Frameworks

### [Pytorch](../api/pytorch)

Integration for PyTorch-based models and experiments.

- Internal docs: [PyTorch API docs](../api/pytorch)
- External docs: [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- External docs: [torchvision documentation](https://pytorch.org/vision/stable/index.html)

```{toctree}
:hidden:

../api/anjana
../api/fairlearn
../api/lifelines
../api/seaborn
../api/yellowbrick
../api/pytorch

```
