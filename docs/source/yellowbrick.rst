Yellowbrick Visualization
=========================

Deckard provides integration with the Yellowbrick library for machine learning
visualization through the :mod:`deckard.plot.yellowbrick_plots` module. This
enables interactive, interpretable visualizations of model behavior and feature
importance.

.. _yellowbrick-overview:

Overview
--------

The :mod:`deckard.plot.yellowbrick_plots` module wraps Yellowbrick visualizers
with Deckard-aware model and data handling:

- **Feature analysis**: feature importance, correlation heatmaps, PCA projection
- **Model evaluation**: learning curves, validation curves, residual plots
- **Classification**: ROC/AUC, confusion matrices, precision-recall curves
- **Regression**: residuals, predictions vs. actual, R² scoring
- **Clustering**: silhouette scores, elbow plots
- **Text analysis**: token frequency, t-SNE projection

Key Features
~~~~~~~~~~~~

- **Integrated with ExperimentConfig**: visualizers operate directly on trained
  models and datasets
- **Statistical interpretation**: overlays confidence intervals, p-values, and
  diagnostic information
- **Interactive plots**: embed in Jupyter notebooks or save as static images
- **Customizable styling**: matplotlib-backed theming and color palettes
- **Model-agnostic**: works with sklearn, PyTorch, and other Deckard-supported models

Limitations
~~~~~~~~~~~

**Important**: Yellowbrick attack visualizers are **not yet supported** in Deckard.
Visualizers that specifically analyze attack success, adversarial perturbations,
or robustness metrics must use Seaborn (see :doc:`seaborn`) or custom matplotlib.

Supported visualization contexts include:

- Standard model training and evaluation
- Feature analysis and data exploration
- Learning and validation curves for hyperparameter tuning
- Cross-validation score distributions

Usage
-----

Command-line examples
~~~~~~~~~~~~~~~~~~~~~

**Basic model evaluation with Yellowbrick:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data.dataset_name=make_classification \
      model.model_type=sklearn.ensemble.RandomForestClassifier \
      file.output_dir=/tmp/exp1

   python -m deckard plot \
      --config-name yellowbrick-plot \
      plot.visualizer_type=learning_curve \
      plot.input_file=/tmp/exp1/scores.pkl \
      plot.output_file=/tmp/exp1/learning_curve.png

**Feature importance visualization:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data.dataset_name=load_iris \
      model.model_type=sklearn.ensemble.RandomForestClassifier

   python -m deckard plot \
      --config-name yellowbrick-plot \
      plot.visualizer_type=feature_importance \
      plot.top_features=10 \
      plot.output_file=/tmp/feature_importance.png

**Confusion matrix with Yellowbrick:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data.dataset_name=make_classification \
      model.model_type=sklearn.ensemble.RandomForestClassifier

   python -m deckard plot \
      --config-name yellowbrick-plot \
      plot.visualizer_type=confusion_matrix \
      plot.normalized=true \
      plot.output_file=/tmp/confusion_matrix.png

Programmatic examples
~~~~~~~~~~~~~~~~~~~~~

**Learning curve visualization:**

.. code-block:: python

   from yellowbrick.model_selection import LearningCurve
   from sklearn.model_selection import StratifiedKFold
   from deckard.experiment import ExperimentConfig
   from deckard.data import DataConfig
   from deckard.model import ModelConfig
   from deckard.score import DefaultClassifierConfig
   import matplotlib.pyplot as plt

   # Run experiment
   data = DataConfig(
       dataset_name="make_classification",
       data_params={"n_samples": 500, "n_features": 20},
       classifier=True,
       scorer=DefaultClassifierConfig(),
   )
   model = ModelConfig(
       model_type="sklearn.ensemble.RandomForestClassifier",
       classifier=True,
       scorer=DefaultClassifierConfig(),
   )
   cfg = ExperimentConfig(data=data, model=model)
   scores = cfg()

   # Create learning curve visualizer
   cv = StratifiedKFold(n_splits=5)
   visualizer = LearningCurve(
       model.get_model(),
       cv=cv,
       scoring="accuracy",
       train_sizes=np.linspace(0.1, 1.0, 10),
   )

   visualizer.fit(data.X_train, data.y_train)
   visualizer.poof(outpath="/tmp/learning_curve.png")

**Feature importance with Yellowbrick:**

.. code-block:: python

   from yellowbrick.model_selection import FeatureImportances
   from deckard.experiment import ExperimentConfig
   from deckard.data import DataConfig
   from deckard.model import ModelConfig
   from deckard.score import DefaultClassifierConfig
   import matplotlib.pyplot as plt

   # Run experiment
   data = DataConfig(
       dataset_name="load_iris",
       classifier=True,
       scorer=DefaultClassifierConfig(),
   )
   model = ModelConfig(
       model_type="sklearn.ensemble.RandomForestClassifier",
       classifier=True,
       scorer=DefaultClassifierConfig(),
   )
   cfg = ExperimentConfig(data=data, model=model)
   scores = cfg()

   # Visualize feature importance
   visualizer = FeatureImportances(
       model.get_model(),
       stack=True,
       feature_names=data.feature_names,
   )
   visualizer.fit(data.X_train, data.y_train)
   visualizer.poof(outpath="/tmp/feature_importance.png")

**Confusion matrix with Yellowbrick:**

.. code-block:: python

   from yellowbrick.classifier import ConfusionMatrix
   from deckard.experiment import ExperimentConfig
   import matplotlib.pyplot as plt

   # Run experiment (see above)
   cfg = ExperimentConfig(data=data, model=model)
   scores = cfg()

   # Get trained model and test data
   trained_model = model.get_model()

   # Visualize confusion matrix
   visualizer = ConfusionMatrix(
       trained_model,
       classes=model.get_classes(),
       normalized=False,
   )
   visualizer.fit(data.X_train, data.y_train)
   visualizer.score(data.X_test, data.y_test)
   visualizer.poof(outpath="/tmp/confusion_matrix.png")

**Operating on ExperimentConfig Output**

Yellowbrick visualizers work naturally with Deckard ExperimentConfig objects:

.. code-block:: python

   from yellowbrick.model_selection import ValidationCurve
   import numpy as np

   # Assume cfg is a trained ExperimentConfig
   model_obj = cfg.model  # the trained model
   X_train, y_train = cfg.data.X_train, cfg.data.y_train

   # Visualize validation curves across hyperparameters
   visualizer = ValidationCurve(
       model_obj,
       param_name="max_depth",
       param_range=np.arange(1, 11),
       cv=5,
       scoring="accuracy",
   )

   visualizer.fit(X_train, y_train)
   visualizer.poof(outpath="/tmp/validation_curve.png")

Configuration
~~~~~~~~~~~~~

Key configuration options for Yellowbrick visualizers:

- **visualizer_type** (str): type of visualizer (learning_curve,
  feature_importance, confusion_matrix, roc_auc, residuals, etc.)
- **input_file** (str): path to scored experiment output
- **output_file** (str): path to save visualization image
- **normalized** (bool): normalize plots (e.g., confusion matrix percentages)
- **top_features** (int): limit feature importance to top N features
- **figure_size** (tuple): matplotlib figure dimensions
- **dpi** (int): resolution for saved images (default: 300)
- **show_legend** (bool): include legend on visualization
- **title** (str): custom title for visualization
- **colors** (list): custom color palette for visualization

Supported Visualizers
~~~~~~~~~~~~~~~~~~~~~

Common Yellowbrick visualizers supported in Deckard context:

- **learning_curve**: training vs. validation score across sample sizes
- **validation_curve**: model performance across a hyperparameter range
- **feature_importance**: relative importance of features in tree-based models
- **confusion_matrix**: predicted vs. actual class counts
- **roc_auc**: receiver operating characteristic curve with AUC
- **precision_recall**: precision vs. recall curve
- **residuals**: regression residual distribution and linearity
- **silhouette**: clustering silhouette scores and distances
- **elbow**: elbow curve for optimal cluster number selection

Export Formats
~~~~~~~~~~~~~~

Yellowbrick supports multiple output formats via ``poof(outpath=...)``

- **PNG** (.png): raster format, good for presentations
- **PDF** (.pdf): vector format, suitable for publications
- **SVG** (.svg): scalable vector, editable in design tools
- **EPS** (.eps): encapsulated PostScript

Use high **dpi** values for publication-quality output.

Troubleshooting
~~~~~~~~~~~~~~~

- **Import error**: ensure yellowbrick is installed via
  ``pip install "deckard[plot,yellowbrick]"`` or ``pip install yellowbrick``
- **Attack visualizers not supported**: use Seaborn instead (see :doc:`seaborn`)
- **Visualization not displayed**: check output_file path is writable; ensure
  backend is set via ``matplotlib.use('Agg')`` for headless environments
- **Memory issues with large datasets**: reduce sample size or use subsampling
  before visualizer fitting
- **No feature names**: Yellowbrick requires feature_names for interpretability;
  pass via config or data object

See also
~~~~~~~~

* :doc:`plot` — general plotting documentation including yellowbrick support
* :doc:`seaborn` — alternative statistical visualization framework
* :doc:`model` — model configuration for visualizer input
* :doc:`data` — data configuration and feature handling
* :doc:`experiment` — experiment orchestration that generates model/data pairs
