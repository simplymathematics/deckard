import pandas as pd

from deckard.data.base import DataConfig
from deckard.data.stages import normalize_data_score_stage


def _cfg_with_loaded_splits() -> DataConfig:
    cfg = DataConfig(dataset_name="make_classification", scorer=lambda **kwargs: {"base": 1})
    cfg._X = pd.DataFrame({"f": [1, 2, 3, 4]})
    cfg._y = pd.Series([0, 1, 0, 1])
    cfg.X_train = pd.DataFrame({"f": [1, 2]})
    cfg.y_train = pd.Series([0, 1])
    cfg.X_test = pd.DataFrame({"f": [3, 4]})
    cfg.y_test = pd.Series([0, 1])
    cfg.X_val = pd.DataFrame({"f": [5]})
    cfg.y_val = pd.Series([1])
    return cfg


def test_normalize_data_score_stage_aliases():
    assert normalize_data_score_stage("post-defense") == "test"
    assert normalize_data_score_stage("attack") == "test"
    assert normalize_data_score_stage("attack-val") == "val"
    assert normalize_data_score_stage("train") == "train"


def test_score_stage_hooks_support_stage_specific_and_legacy_hooks():
    seen = {"before": 0, "after": 0, "legacy_after": 0}

    class Plugin:
        def before_score_test(self, runtime, **kwargs):
            _ = runtime
            seen["before"] += 1
            assert kwargs["stage"] == "test"
            return None

        def after_score_test(self, runtime, **kwargs):
            _ = runtime
            seen["after"] += 1
            assert kwargs["stage"] == "test"
            return {"stage_specific": 1}

        def after_score(self, runtime, **kwargs):
            _ = runtime
            seen["legacy_after"] += 1
            assert kwargs["stage"] == "test"
            return {"legacy": 2}

    cfg = _cfg_with_loaded_splits()
    cfg.plugins = [Plugin()]
    cfg._plugin_objects = cfg.plugins

    result = cfg.score(mode="test")

    assert result["base"] == 1
    assert result["stage_specific"] == 1
    assert result["legacy"] == 2
    assert seen == {"before": 1, "after": 1, "legacy_after": 1}


def test_score_mode_alias_routes_to_canonical_stage_hooks():
    called = {"after_test": 0}

    class Plugin:
        def after_score_test(self, runtime, **kwargs):
            _ = runtime
            assert kwargs["stage"] == "test"
            called["after_test"] += 1
            return {"ok": True}

    cfg = _cfg_with_loaded_splits()
    cfg.plugins = [Plugin()]
    cfg._plugin_objects = cfg.plugins

    result = cfg.score(mode="post-defense")

    assert result["ok"] is True
    assert called["after_test"] == 1
