"""Named score-profile declarations and ConfigStore registrations."""

from pathlib import Path
from omegaconf import OmegaConf
from .base import DefaultModelScorerConfig, ScorerTypePlugin, safe_store
from .data import DefaultDataScorerConfig
from .fairness import DefaultFairlearnScorerConfig


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


# Register scorer plugins to ConfigStore
safe_store(group="score/plugins", name="model", node=SCORER_PLUGIN_MODEL_BASE)
safe_store(group="score/plugins", name="model-classifier", node=SCORER_PLUGIN_MODEL_CLASSIFIER)
safe_store(group="score/plugins", name="model-regressor", node=SCORER_PLUGIN_MODEL_REGRESSOR)
safe_store(group="score/plugins", name="data", node=SCORER_PLUGIN_DATA_BASE)
safe_store(group="score/plugins", name="data-classifier", node=SCORER_PLUGIN_DATA_CLASSIFIER)
safe_store(group="score/plugins", name="data-regressor", node=SCORER_PLUGIN_DATA_REGRESSOR)
safe_store(group="score/plugins", name="fairness", node=SCORER_PLUGIN_FAIRNESS)

# Register same plugins to search/score/plugins for search space composition
safe_store(group="search/score/plugins", name="model", node=SCORER_PLUGIN_MODEL_BASE)
safe_store(group="search/score/plugins", name="model-classifier", node=SCORER_PLUGIN_MODEL_CLASSIFIER)
safe_store(group="search/score/plugins", name="model-regressor", node=SCORER_PLUGIN_MODEL_REGRESSOR)
safe_store(group="search/score/plugins", name="data", node=SCORER_PLUGIN_DATA_BASE)
safe_store(group="search/score/plugins", name="data-classifier", node=SCORER_PLUGIN_DATA_CLASSIFIER)
safe_store(group="search/score/plugins", name="data-regressor", node=SCORER_PLUGIN_DATA_REGRESSOR)
safe_store(group="search/score/plugins", name="fairness", node=SCORER_PLUGIN_FAIRNESS)


def _load_example_score_configs():
    """Load score configs from examples/sklearn/config/score and register with ConfigStore."""
    examples_dir = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "sklearn"
        / "config"
        / "score"
    )

    if not examples_dir.exists():
        return

    for yaml_file in sorted(examples_dir.glob("*.yaml")):
        try:
            config_name = yaml_file.stem
            cfg = OmegaConf.load(yaml_file)
            safe_store(group="score", name=config_name, node=cfg)
        except Exception:
            pass  # Silently skip any problematic configs


_load_example_score_configs()
