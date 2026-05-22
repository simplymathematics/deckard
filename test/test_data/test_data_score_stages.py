import pandas as pd
import pytest

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
    assert normalize_data_score_stage("train") == "train"
    assert normalize_data_score_stage("test") == "test"
    assert normalize_data_score_stage("val") == "val"
    assert normalize_data_score_stage("all") == "all"
    with pytest.raises(ValueError):
        normalize_data_score_stage("post-defense")


def test_score_is_pass_through_to_scorer_dict_config():
    captured = {}

    class _CaptureScorer:
        scoring_type = "data"

        def __call__(self, *args, **kwargs):
            _ = args
            captured.update(kwargs)
            return {"ok": True}

    cfg = _cfg_with_loaded_splits()
    cfg.scorer = _CaptureScorer()

    result = cfg.score(mode="test", stage="post-sample")

    assert result["ok"] is True
    assert captured["mode"] == "test"
    assert captured["stage"] == "post-sample"
    assert captured["data"] is cfg


def test_call_orchestrates_scores_using_scorer_stages():
    cfg = DataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 4,
            "n_informative": 2,
            "n_redundant": 0,
            "random_state": 42,
            "n_clusters_per_class": 1,
        },
        score_split="test",
        scorer={
            "_target_": "deckard.score.data.DefaultDataScorerConfig",
            "scorers": {
                "pre_metric": {
                    "score_function": lambda y_true, y_pred: float(len(y_true)),
                    "stage": "post-sample",
                },
                "post_metric": {
                    "score_function": lambda y_true, y_pred: float(len(y_true)),
                    "stage": "post-pipeline",
                },
            },
        },
    )

    out = cfg()

    assert "test" in out
    assert "pre_metric" in out["test"]
    assert "post_metric" in out["test"]
