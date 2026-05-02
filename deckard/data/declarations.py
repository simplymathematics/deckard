"""Static data configuration declarations and ConfigStore registrations."""

from ..utils import safe_store
from .sample import register_sampler_configs


# Static data options mirrored from examples/sklearn/config/data.
DATA_ADULT = {
    "dataset_name": "adult",
    "test_size": 0.2,
    "random_state": 42,
    "classifier": True,
    "stratify": False,
    "alias": "adult",
    "sample": "split",
    "pipeline": {
        "imputer": {
            "name": "sklearn.impute.SimpleImputer",
            "strategy": "mean",
        },
    },
}

DATA_DIABETES = {
    "dataset_name": "diabetes",
    "test_size": 0.2,
    "random_state": 42,
    "classifier": False,
    "stratify": False,
    "alias": "diabetes",
    "sample": "split",
}

DATA_CLASSIFICATION = {
    "dataset_name": "make_classification",
    "data_params": {
        "n_samples": 10000,
        "n_features": 20,
        "n_informative": 15,
        "n_redundant": 5,
        "n_classes": 2,
        "random_state": 42,
    },
    "test_size": 0.2,
    "random_state": 42,
    "classifier": True,
    "alias": "make_classification",
    "sample": "split",
}

DATA_REGRESSION = {
    "dataset_name": "make_regression",
    "data_params": {
        "n_samples": 10000,
        "n_features": 20,
        "n_informative": 10,
        "noise": 0.1,
        "random_state": 42,
    },
    "test_size": 0.2,
    "random_state": 42,
    "stratify": False,
    "classifier": False,
    "alias": "make_regression",
    "sample": "split",
}


safe_store(group="data", name="adult", node=DATA_ADULT)
safe_store(group="data", name="diabetes", node=DATA_DIABETES)
safe_store(group="data", name="classification", node=DATA_CLASSIFICATION)
safe_store(group="data", name="regression", node=DATA_REGRESSION)

# Register static sampler options from deckard.data.sample.
try:
    register_sampler_configs()
except Exception:
    pass

