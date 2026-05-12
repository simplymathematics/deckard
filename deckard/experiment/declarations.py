"""Static survival ConfigStore declarations used by tests and Hydra overrides."""

from ..utils import safe_store

LIFELINES_DATASETS = {
    "lung": {
        "_target_": "deckard.data.DataConfig",
        "dataset_name": "lifelines.lung",
        "target": None,
        "classifier": False,
        "stratify": False,
    },
    "diabetes": {
        "_target_": "deckard.data.DataConfig",
        "dataset_name": "lifelines_diabetes",
        "target": None,
        "classifier": False,
        "stratify": False,
    },
}

SURVIVAL_MODELS = {
    "weibull": "weibull",
    "cox": "cox",
    "aalen": "aalen",
}


for dataset_name, dataset_cfg in LIFELINES_DATASETS.items():
    safe_store(group="survival/data", name=dataset_name, node=dataset_cfg)

for model_name, model_value in SURVIVAL_MODELS.items():
    safe_store(group="survival/model", name=model_name, node=model_value)
