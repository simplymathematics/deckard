from ..utils import safe_store

# TinyNet config and registration (must be after safe_store import)
TINYNET_MODEL = {
    "model_type": "deckard.model.pytorch.TinyNet",
    "classifier": True,
    "model_params": {
        "input_dim": 10,  # Set default, should be overridden by data shape
        "hidden_dim": 16,
        "output_dim": 2,
    },
    "_target_": "deckard.model.pytorch.PytorchModelConfig",
    "alias": "tinynet",
}
safe_store(group="model", name="tinynet", node=TINYNET_MODEL)
safe_store(group="search/models", name="tinynet", node=TINYNET_MODEL)
"""Static model/defense configuration declarations and ConfigStore registrations."""


"""Static model/defense configuration declarations and ConfigStore registrations."""

# Static model options mirrored from examples/sklearn/config/model.
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

# Static defense options mirrored from examples/sklearn/config/defense.
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

DEFENSE_FEATURE_SQUEEZING = {
    "defense_name": "art.defences.preprocessor.FeatureSqueezing",
    "init_params": {
        "library": "art",
        "type": "preprocessor",
        "class": "FeatureSqueezing",
    },
    "defense_params": {
        "apply_fit": False,
        "apply_predict": True,
        "bit_depth": 8,
        "clip_values": [0, 255],
    },
    "alias": "feature-squeezing",
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

DEFENSE_PLUGIN_DETECTOR = {
    "name": "deckard.model.defend.DefenseTypePlugin",
    "mixin_type": "deckard.model.detector._DetectorDefenseMixin",
    "defense_type": "detector",
    "init_params": {
        "library": "art",
        "type": "defense",
        "class": "detector",
    },
}

DEFENSE_PLUGIN_PREPROCESSOR = {
    "name": "deckard.model.defend.DefenseTypePlugin",
    "mixin_type": "deckard.model.preprocessor._PreprocessorDefenseMixin",
    "defense_type": "preprocessor",
    "init_params": {
        "library": "art",
        "type": "defense",
        "class": "preprocessor",
    },
}

DEFENSE_PLUGIN_POSTPROCESSOR = {
    "name": "deckard.model.defend.DefenseTypePlugin",
    "mixin_type": "deckard.model.postprocessor._PostprocessorDefenseMixin",
    "defense_type": "postprocessor",
    "init_params": {
        "library": "art",
        "type": "defense",
        "class": "postprocessor",
    },
}

DEFENSE_PLUGIN_TRAINER = {
    "name": "deckard.model.defend.DefenseTypePlugin",
    "mixin_type": "deckard.model.trainer._TrainerDefenseMixin",
    "defense_type": "trainer",
    "init_params": {
        "library": "art",
        "type": "defense",
        "class": "trainer",
    },
}

DEFENSE_PLUGIN_TRANSFORMER = {
    "name": "deckard.model.defend.DefenseTypePlugin",
    "mixin_type": "deckard.model.transformer._TransformerDefenseMixin",
    "defense_type": "transformer",
    "init_params": {
        "library": "art",
        "type": "defense",
        "class": "transformer",
    },
}

DEFENSE_PLUGIN_REGULARIZER = {
    "name": "deckard.model.defend.DefenseTypePlugin",
    "mixin_type": "deckard.model.regularizer._RegularizerDefenseMixin",
    "defense_type": "regularizer",
    "init_params": {
        "library": "art",
        "type": "defense",
        "class": "regularizer",
    },
}


safe_store(group="model", name="logistic", node=MODEL_LOGISTIC)
safe_store(group="model", name="rf", node=MODEL_RF)
safe_store(group="model", name="svc", node=MODEL_SVC)
safe_store(group="model", name="ridge", node=MODEL_RIDGE)
safe_store(group="model", name="linear", node=MODEL_LINEAR)

safe_store(group="search/models", name="logistic", node=MODEL_LOGISTIC)
safe_store(group="search/models", name="rf", node=MODEL_RF)
safe_store(group="search/models", name="svc", node=MODEL_SVC)
safe_store(group="search/models", name="ridge", node=MODEL_RIDGE)
safe_store(group="search/models", name="linear", node=MODEL_LINEAR)

safe_store(group="defense", name="baseline", node=DEFENSE_BASELINE)
safe_store(group="defense", name="class-labels", node=DEFENSE_CLASS_LABELS)
safe_store(
    group="defense",
    name="feature-squeezing",
    node=DEFENSE_FEATURE_SQUEEZING,
)
safe_store(group="defense", name="anjana", node=DEFENSE_ANJANA)

safe_store(group="search/defenses", name="baseline", node=DEFENSE_BASELINE)
safe_store(
    group="search/defenses",
    name="class-labels",
    node=DEFENSE_CLASS_LABELS,
)
safe_store(
    group="search/defenses",
    name="feature-squeezing",
    node=DEFENSE_FEATURE_SQUEEZING,
)
safe_store(group="search/defenses", name="anjana", node=DEFENSE_ANJANA)

safe_store(group="defense/plugins", name="detector", node=DEFENSE_PLUGIN_DETECTOR)
safe_store(
    group="defense/plugins",
    name="preprocessor",
    node=DEFENSE_PLUGIN_PREPROCESSOR,
)
safe_store(
    group="defense/plugins",
    name="postprocessor",
    node=DEFENSE_PLUGIN_POSTPROCESSOR,
)
safe_store(group="defense/plugins", name="trainer", node=DEFENSE_PLUGIN_TRAINER)
safe_store(
    group="defense/plugins",
    name="transformer",
    node=DEFENSE_PLUGIN_TRANSFORMER,
)
safe_store(
    group="defense/plugins",
    name="regularizer",
    node=DEFENSE_PLUGIN_REGULARIZER,
)

safe_store(
    group="search/defense/plugins",
    name="detector",
    node=DEFENSE_PLUGIN_DETECTOR,
)
safe_store(
    group="search/defense/plugins",
    name="preprocessor",
    node=DEFENSE_PLUGIN_PREPROCESSOR,
)
safe_store(
    group="search/defense/plugins",
    name="postprocessor",
    node=DEFENSE_PLUGIN_POSTPROCESSOR,
)
safe_store(
    group="search/defense/plugins",
    name="trainer",
    node=DEFENSE_PLUGIN_TRAINER,
)
safe_store(
    group="search/defense/plugins",
    name="transformer",
    node=DEFENSE_PLUGIN_TRANSFORMER,
)
safe_store(
    group="search/defense/plugins",
    name="regularizer",
    node=DEFENSE_PLUGIN_REGULARIZER,
)
