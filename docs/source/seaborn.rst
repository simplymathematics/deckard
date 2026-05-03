Seaborn Visualization
=====================

Deckard provides native support for statistical visualization through Seaborn
integration in the :mod:`deckard.plot.seaborn_plots` module. This enables
publication-quality plots for model performance, fairness metrics, and attack
analysis.

.. _seaborn-overview:

Overview
--------

The :mod:`deckard.plot.seaborn_plots` module wraps common Seaborn plotting
functions with Deckard-aware data handling:

- **Classification metrics**: confusion matrices, ROC curves, precision-recall
- **Regression metrics**: residual plots, prediction vs. actual
- **Fairness analysis**: group-wise metric comparisons via fairlearn integration
- **Attack analysis**: evasion success rates, membership inference accuracy
- **Customization**: matplotlib-based styling with theme support

Seaborn plots integrate seamlessly with :class:`deckard.experiment.ExperimentConfig`
output and :class:`deckard.file.FileConfig` artifact storage.

Key Features
~~~~~~~~~~~~

- **Score-driven plotting**: generate plots directly from scored experiment output
- **Group-aware visualization**: compare metrics across sensitive attributes
  (fairness groups)
- **Multi-run aggregation**: plot results across multiple experiment runs
- **Export formats**: save as PNG, PDF, SVG with publication quality
- **Customizable themes**: switch between Seaborn palettes (darkgrid, whitegrid,
  dark, white, ticks)

Plot Types
~~~~~~~~~~

Common Seaborn-based plots include:

- **confusionmatrix**: heatmap of predicted vs. actual classes
- **rocauc**: receiver operating characteristic curve
- **precisionrecall**: precision vs. recall curve
- **residuals**: regression residual distribution
- **fairness_comparison**: group-wise metric barplot with error bars
- **attack_success**: membership/attribute inference success rates by group
- **training_history**: loss/accuracy curves across epochs

Usage
-----

Command-line examples
~~~~~~~~~~~~~~~~~~~~~

**Plot model performance with Seaborn:**

.. code-block:: bash

   # Run experiment and save scores
   python -m deckard optimize --config-name experiment \
      data.dataset_name=make_classification \
      model.model_type=sklearn.ensemble.RandomForestClassifier \
      file.output_dir=/tmp/exp1

   # Generate confusion matrix plot
   python -m deckard plot \
      --config-name seaborn-plot \
      plot.plot_type=confusionmatrix \
      plot.input_file=/tmp/exp1/scores.pkl \
      plot.output_file=/tmp/exp1/confusion_matrix.png

**Plot fairness metrics:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data=fairness \
      data.sensitive_feature=gender \
      model.model_type=sklearn.ensemble.RandomForestClassifier

   python -m deckard plot \
      --config-name seaborn-plot \
      plot.plot_type=fairness_comparison \
      plot.group_by=gender \
      plot.metrics=[accuracy,precision,recall] \
      plot.output_file=/tmp/fairness_comparison.png

**Plot attack results:**

.. code-block:: bash

   python -m deckard optimize --config-name experiment \
      data.dataset_name=make_classification \
      model.model_type=sklearn.ensemble.RandomForestClassifier \
      attack.attack_type=art.attacks.evasion.FastGradientMethod

   python -m deckard plot \
      --config-name seaborn-plot \
      plot.plot_type=attack_success \
      plot.input_file=/tmp/exp1/scores.pkl \
      plot.output_file=/tmp/exp1/attack_results.png

Programmatic examples
~~~~~~~~~~~~~~~~~~~~~

**Generate confusion matrix:**

.. code-block:: python

   import matplotlib.pyplot as plt
   from deckard.plot.seaborn_plots import confusion_matrix_plot
   from deckard.experiment import ExperimentConfig
   from deckard.data import DataConfig
   from deckard.model import ModelConfig
   from deckard.score import DefaultClassifierConfig

   # Run experiment
   data = DataConfig(
       dataset_name="make_classification",
       data_params={"n_samples": 200, "n_features": 10},
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

   # Plot confusion matrix
   plt.figure(figsize=(8, 6))
   confusion_matrix_plot(scores["predictions"], scores["y_test"])
   plt.title("Model Performance: Confusion Matrix")
   plt.tight_layout()
   plt.savefig("confusion_matrix.png", dpi=300)
   plt.close()

**Plot fairness comparison with group breakdown:**

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   from deckard.data.fairness import FairlearnDataConfig
   from deckard.score.fairness import FairlearnScoreDictConfig

   # Assume scores dict has "fairness_group_metrics" with {group: {metric: value}}
   fairness_scores = scores.get("fairness_group_metrics", {})

   # Prepare data for Seaborn barplot
   groups = []
   metrics_list = []
   values = []
   for group, metrics in fairness_scores.items():
       for metric, value in metrics.items():
           groups.append(group)
           metrics_list.append(metric)
           values.append(value)

   import pandas as pd
   df = pd.DataFrame({
       "Group": groups,
       "Metric": metrics_list,
       "Value": values,
   })

   plt.figure(figsize=(10, 6))
   sns.barplot(data=df, x="Metric", y="Value", hue="Group)
   plt.title("Fairness Metrics by Group")
   plt.ylabel("Metric Value")
   plt.tight_layout()
   plt.savefig("fairness_comparison.png", dpi=300)
   plt.close()

**Plot ROC curve:**

.. code-block:: python

   import matplotlib.pyplot as plt
   from sklearn.metrics import roc_curve, auc
   from deckard.plot.seaborn_plots import roc_plot

   # Extract predictions and labels from scores
   y_true = scores["y_test"]
   y_pred = scores["y_pred_proba"][:, 1]  # positive class probability

   fpr, tpr, thresholds = roc_curve(y_true, y_pred)
   roc_auc = auc(fpr, tpr)

   plt.figure(figsize=(8, 6))
   plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
   plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
   plt.xlabel("False Positive Rate")
   plt.ylabel("True Positive Rate")
   plt.title("Receiver Operating Characteristic")
   plt.legend()
   plt.tight_layout()
   plt.savefig("roc_curve.png", dpi=300)
   plt.close()

Configuration
~~~~~~~~~~~~~

Key configuration options for Seaborn plots:

- **plot_type** (str): plot style (confusionmatrix, rocauc, residuals, etc.)
- **input_file** (str): path to scored experiment output (.pkl, .json)
- **output_file** (str): path to save plot image
- **figure_size** (tuple): matplotlib figure size (width, height)
- **dpi** (int): resolution for saved image (default: 300)
- **palette** (str): Seaborn color palette (Set2, husl, dark, pastel, etc.)
- **style** (str): Seaborn style (darkgrid, whitegrid, dark, white, ticks)
- **group_by** (str): column name for grouping (e.g., sensitive feature for fairness)
- **metrics** (list): specific metrics to plot
- **show_legend** (bool): include legend on plot
- **title** (str): custom plot title
- **xlabel**, **ylabel** (str): axis labels

Export Formats
~~~~~~~~~~~~~~

Seaborn plots support multiple output formats:

- **PNG** (.png): raster format, good for presentations and web
- **PDF** (.pdf): vector format, suitable for papers and publications
- **SVG** (.svg): scalable vector, editable in design tools
- **EPS** (.eps): encapsulated PostScript, legacy publication format

Use the **dpi** parameter to control raster resolution; higher DPI produces
sharper images but larger file sizes.

Styling
~~~~~~~

Customize Seaborn aesthetics:

.. code-block:: python

   import seaborn as sns

   # Set style for all plots
   sns.set_style("whitegrid")  # clean background with gridlines
   sns.set_palette("Set2")  # colorblind-friendly palette

   # Or specify per-plot via plot config
   plot_config = {
       "style": "whitegrid",
       "palette": "husl",
       "figure_size": (12, 8),
   }

Troubleshooting
~~~~~~~~~~~~~~~

- **Import error**: ensure seaborn is installed via ``pip install "deckard[plot]"``
- **Empty plots**: verify input_file contains valid score dictionary with expected
  keys
- **File not found**: check output_file directory exists and is writable
- **Poor figure quality**: increase dpi parameter or use PDF/SVG format
- **Legend crowded**: reduce figure_size or use ``show_legend=False``

See also
~~~~~~~~

* :doc:`plot` — general plotting documentation including seaborn support
* :doc:`data` — data configuration for plotting input
* :doc:`score` — scoring framework that produces plotting data
* :doc:`experiment` — experiment orchestration that generates scores
* :doc:`yellowbrick` — alternative visualization framework
