"""Model configuration module.

This module is kept for backward compatibility.
Canonical model and defense configs are now loaded from examples/*/config/ YAML files
at runtime via deckard.declarations.register_configs().

Previous static ConfigStore registrations have been consolidated into YAML files:
- model configs: examples/sklearn/config/model/ (logistic, rf, svc, ridge, linear, ...)
- model configs: examples/pytorch/config/model/ (tinynet, ...)
- defense configs: examples/sklearn/config/defense/ (baseline, class-labels, anjana, ...)

Reference dictionaries are kept below for documentation only.
"""

# Reference dictionaries for documentation (no longer registered via safe_store)

TINYNET_MODEL = {
    "model_type": "deckard.pytorch.model.TinyNet",
    "classifier": True,
    "model_params": {
        "input_dim": 10,  # Set default, should be overridden by data shape
        "hidden_dim": 16,
        "output_dim": 2,
    },
    "_target_": "deckard.pytorch.model.PytorchModelConfig",
    "alias": "tinynet",
}

MODEL_LOGISTIC = {
    "model_type": "sklearn.linear_model.LogisticRegression",
    "classifier": True,
    "model_params": {
        "penalty": "l2",
        "dual": False,
        "tol": 0.0001,
        "C": 1.0,
        "fit_intercept": True,
        "max_iter": 10,
    },
    "_target_": "deckard.model.ModelConfig",
    "alias": "logistic",
}

MODEL_RF = {
    "model_type": "sklearn.ensemble.RandomForestClassifier",
    "classifier": True,
    "model_params": {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
    },
    "_target_": "deckard.model.ModelConfig",
    "alias": "rf",
}

MODEL_SVC = {
    "model_type": "sklearn.svm.SVC",
    "classifier": True,
    "model_params": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "coef0": 0.0,
        "shrinking": True,
        "probability": True,
        "tol": 0.001,
        "cache_size": 200,
        "class_weight": None,
        "decision_function_shape": "ovr",
        "break_ties": False,
        "random_state": None,
        "max_iter": 100,
        "verbose": False,
    },
    "_target_": "deckard.model.ModelConfig",
    "alias": "svc",
}

MODEL_RIDGE = {
    "model_type": "sklearn.linear_model.Ridge",
    "classifier": False,
    "model_params": {
        "tol": 0.0001,
        "fit_intercept": True,
        "alpha": 1.0,
    },
    "_target_": "deckard.model.ModelConfig",
    "alias": "ridge",
}

MODEL_LINEAR = {
    "model_type": "sklearn.linear_model.LinearRegression",
    "classifier": False,
    "model_params": {
        "tol": 0.0001,
        "fit_intercept": True,
    },
    "_target_": "deckard.model.ModelConfig",
    "alias": "linear",
}

DEFENSE_BASELINE = {
    "defense_name": None,
    "init_params": {
        "library": "art",
        "type": "defense",
        "class": "baseline",
    },
    "defense_params": {},
    "_target_": "deckard.DefenseConfig",
    "alias": "baseline",
}

DEFENSE_CLASS_LABELS = {
    "defense_name": "art.defences.postprocessor.ClassLabels",
    "init_params": {
        "library": "art",
        "type": "postprocessor",
        "class": "ClassLabels",
    },
    "defense_params": {
        "apply_fit": False,
        "apply_predict": True,
    },
    "_target_": "deckard.DefenseConfig",
    "alias": "class-labels",
}

DEFENSE_ANJANA = {
    "defense_name": None,
    "init_params": {
        "library": "anjana",
        "type": "data",
        "class": "anonymization",
    },
    "defense_params": {
        "name": "anjana.anonymity.k_anonymity",
        "k": 2,
    },
    "_target_": "deckard.DefenseConfig",
    "alias": "anjana",
}

__all__ = [
    "TINYNET_MODEL",
    "MODEL_LOGISTIC",
    "MODEL_RF",
    "MODEL_SVC",
    "MODEL_RIDGE",
    "MODEL_LINEAR",
    "DEFENSE_BASELINE",
    "DEFENSE_CLASS_LABELS",
    "DEFENSE_ANJANA",
]
