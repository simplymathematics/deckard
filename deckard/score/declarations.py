"""Score configuration module.

This module is kept for backward compatibility.
Canonical score configs are now loaded from examples/*/config/score/ YAML files
at runtime via deckard.declarations.register_configs().

Reference dictionaries are kept below for documentation only.
"""

from pathlib import Path
from omegaconf import OmegaConf
from .base import DefaultModelScorerConfig, ScorerTypePlugin
from .data import DefaultDataScorerConfig


def _load_example_score_configs() -> None:
    """Backward-compatible no-op loader.

    Score configs are registered dynamically from YAML via
    ``deckard.declarations.register_configs()`` at package import time.
    This function remains to preserve older call sites and tests.
    """

    return None


# Scorer Plugin Declarations
# ==========================
# These plugins contribute mixins to scorer runtime context assembly
# based on scoring_type and optional scoring_subtype.
# This follows the _Mixin -> Plugin -> Config pattern consistent with
# attack and defense plugins.

SCORER_PLUGIN_MODEL_BASE = {
    "name": "deckard.score.base.ScorerTypePlugin",
    "mixin_type": "deckard.score.base._ScorerMixin",
    "scoring_type": "model",
    "init_params": {
        "library": "deckard",
        "type": "scorer",
        "class": "model",
    },
}

SCORER_PLUGIN_MODEL_CLASSIFIER = {
    "name": "deckard.score.base.ScorerTypePlugin",
    "mixin_type": "deckard.score.base._ScorerMixin",
    "scoring_type": "model",
    "scoring_subtype": "classifier",
    "init_params": {
        "library": "deckard",
        "type": "scorer",
        "class": "model.classifier",
    },
}

SCORER_PLUGIN_MODEL_REGRESSOR = {
    "name": "deckard.score.base.ScorerTypePlugin",
    "mixin_type": "deckard.score.base._ScorerMixin",
    "scoring_type": "model",
    "scoring_subtype": "regressor",
    "init_params": {
        "library": "deckard",
        "type": "scorer",
        "class": "model.regressor",
    },
}

SCORER_PLUGIN_DATA_BASE = {
    "name": "deckard.score.base.ScorerTypePlugin",
    "mixin_type": "deckard.score.base._ScorerMixin",
    "scoring_type": "data",
    "init_params": {
        "library": "deckard",
        "type": "scorer",
        "class": "data",
    },
}

SCORER_PLUGIN_DATA_CLASSIFIER = {
    "name": "deckard.score.base.ScorerTypePlugin",
    "mixin_type": "deckard.score.base._ScorerMixin",
    "scoring_type": "data",
    "scoring_subtype": "classifier",
    "init_params": {
        "library": "deckard",
        "type": "scorer",
        "class": "data.classifier",
    },
}

SCORER_PLUGIN_DATA_REGRESSOR = {
    "name": "deckard.score.base.ScorerTypePlugin",
    "mixin_type": "deckard.score.base._ScorerMixin",
    "scoring_type": "data",
    "scoring_subtype": "regressor",
    "init_params": {
        "library": "deckard",
        "type": "scorer",
        "class": "data.regressor",
    },
}

SCORER_PLUGIN_FAIRNESS = {
    "name": "deckard.score.base.ScorerTypePlugin",
    "mixin_type": "deckard.score.base._ScorerMixin",
    "scoring_type": "model",
    "scoring_subtype": "fairness",
    "init_params": {
        "library": "fairlearn",
        "type": "scorer",
        "class": "fairness",
    },
}

# Configs are now loaded from YAML files in examples/*/config/score/
# These dictionaries are kept for reference/legacy code but not registered via safe_store

__all__ = [
    "_load_example_score_configs",
    "SCORER_PLUGIN_MODEL_BASE",
    "SCORER_PLUGIN_MODEL_CLASSIFIER",
    "SCORER_PLUGIN_MODEL_REGRESSOR",
    "SCORER_PLUGIN_DATA_BASE",
    "SCORER_PLUGIN_DATA_CLASSIFIER",
    "SCORER_PLUGIN_DATA_REGRESSOR",
    "SCORER_PLUGIN_FAIRNESS",
    "DefaultModelScorerConfig",
    "DefaultDataScorerConfig",
]


