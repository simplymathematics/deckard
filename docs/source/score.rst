
score.py
========

This module provides a flexible and extensible framework for scoring machine learning models in Deckard.
It defines configuration classes for individual scoring metrics and collections of metrics, supporting both classification and regression tasks.

Classes
-------

ScorerConfig
------------
    A data class that encapsulates a scoring function, its name, and runtime parameters.
    Ensures the scoring function is callable and provides a unified interface for computing scores.
    
    Attributes
    ----------
    score_name : str
        The name of the scoring metric.
    score_function : callable
        The function used to compute the score.
    score_params : dict, optional
        Additional parameters for the scoring function.
    greater_is_better : bool
        Indicates if higher scores are better (default: True).
    needs_proba : bool
        Indicates if the scorer requires probability estimates (default: False).

    Methods
    -------
    __post_init__()
        Validates the scoring function and initializes parameters.
    __call__(y_true, y_pred, swap=False, **kwargs)
        Computes the score using the configured function and parameters.

ScorerDictConfig
----------------
    Manages a dictionary of ScorerConfig instances for batch scoring.
    Allows application of multiple scoring metrics to predictions and ground truth labels.

    Parameters
    ----------
    scorers : dict
        Dictionary mapping metric names to ScorerConfig instances.

    Methods
    -------
    __post_init__()
        Validates all values in the scorers dictionary.
    __iter__()
        Iterates over (key, ScorerConfig) pairs.
    __getitem__(key)
        Retrieves a ScorerConfig by key.
    get_callables()
        Returns a dictionary of all available scoring functions.
    __call__(y_true, y_pred, **kwargs)
        Computes and returns a dictionary of scores for the given inputs.

DefaultClassifierDict
---------------------
    Provides a standardized set of scoring metrics for classification tasks.

    Attributes
    ----------
    scorers : ScorerDictConfig
        Contains common classification scorers:
            - "accuracy": Overall accuracy.
            - "precision": Weighted precision, zero_division=0.
            - "recall": Weighted recall, zero_division=0.
            - "f1": Weighted F1 score, zero_division=0.
            - "roc_auc": Weighted ROC AUC, multi-class 'ovr'.
            - "log_loss": Log loss with optional label specification.

    Usage
    -----
    Used to evaluate classification models with a consistent set of metrics.

DefaultRegressorDict
--------------------
    Provides a standardized set of scoring metrics for regression tasks.

    Attributes
    ----------
    scorers : ScorerDictConfig
        Contains common regression scorers:
            - "mse": Mean Squared Error.
            - "mae": Mean Absolute Error.
            - "r2": R^2 Score.

    Usage
    -----
    Used to evaluate regression models with a consistent set of metrics.

Dependencies
------------
- scikit-learn metrics (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, log_loss)
- Python dataclasses
- Logging
- Deckard's ConfigBase
