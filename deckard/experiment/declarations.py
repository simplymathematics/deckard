"""Experiment configuration module.

This module is kept for backward compatibility.
Canonical experiment configs are now loaded from examples/*/config/experiment/ YAML files
at runtime via deckard.declarations.register_configs().

Reference dictionaries are kept below for documentation only.
"""

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

# Configs are now loaded from YAML files in examples/*/config/experiment/
# These dictionaries are kept for reference/legacy code but not registered via safe_store

__all__ = ["LIFELINES_DATASETS", "SURVIVAL_MODELS"]
