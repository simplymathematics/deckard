import pandas as pd
import pytest
from omegaconf import ListConfig

from deckard.data.base import DataConfig
from deckard.plugins.fairlearn.data import FairlearnDataConfig


def _cfg():
    cfg = FairlearnDataConfig.__new__(FairlearnDataConfig)
    cfg.pipeline = {}
    cfg.sensitive_columns = ["group"]
    cfg.fairness_defense = None
    cfg.scorer = None
    cfg.classifier = True
    return cfg


def test_post_init_converts_sensitive_listconfig(monkeypatch):
    cfg = _cfg()
    cfg.sensitive_columns = ListConfig(["group"])

    monkeypatch.setattr(DataConfig, "__post_init__", lambda self: None)
    monkeypatch.setattr(FairlearnDataConfig, "_validate_init", lambda self: None)

    cfg.__post_init__()

    assert cfg.sensitive_columns == ["group"]


def test_sensitive_label_builder_and_validation_errors():
    cfg = _cfg()
    frame = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})

    cfg.sensitive_columns = None
    with pytest.raises(ValueError, match="must be configured"):
        cfg._sensitive_labels_from_frame(frame)

    cfg.sensitive_columns = ["a", "b"]
    labels = cfg._sensitive_labels_from_frame(frame)
    assert tuple(labels.iloc[0]) == ("x", "1")

    with pytest.raises(ValueError, match="empty"):
        cfg._validate_sensitive_runtime([], "ctx")
    with pytest.raises(ValueError, match="all null"):
        cfg._validate_sensitive_runtime([None, None], "ctx")
    with pytest.raises(ValueError, match="blank"):
        cfg._validate_sensitive_runtime([" ", ""], "ctx")


def test_apply_defense_branch_paths():
    cfg = _cfg()

    cfg.fairness_defense = True
    with pytest.raises(ValueError, match="ambiguous"):
        cfg.apply_defense()

    cfg.fairness_defense = 7
    with pytest.raises(TypeError, match="must be a dict"):
        cfg.apply_defense()

    cfg.fairness_defense = {"name": "fairlearn.preprocessing.CorrelationRemover"}
    cfg._X = None
    cfg.apply_defense()

    cfg._X = pd.DataFrame({"group": ["a", "b"], "x": [1, 2]})
    cfg.sensitive_columns = None
    with pytest.raises(ValueError, match="must be configured"):
        cfg.apply_defense()

    cfg.sensitive_columns = ["missing"]
    with pytest.raises(RuntimeError, match="Sensitive features not found"):
        cfg.apply_defense()

    cfg.sensitive_columns = ["group"]
    cfg.fairness_defense = {}
    with pytest.raises(ValueError, match="include a 'name' key"):
        cfg.apply_defense()

    cfg.fairness_defense = {
        "name": "fairlearn.preprocessing.CorrelationRemover",
        "step_name": "fairness_step",
    }
    cfg.pipeline = {"existing": {"name": "noop"}}
    cfg.apply_defense()
    assert "fairness_step" in cfg.pipeline

    before = dict(cfg.pipeline)
    cfg.apply_defense()
    assert cfg.pipeline == before

    cfg.fairness_defense = {
        "name": "fairlearn.preprocessing.PrototypeRepresentationLearner",
        "step_name": "prototype_step",
        "max_iter": 1,
    }
    cfg.pipeline = {}
    cfg.apply_defense()
    assert "prototype_step" in cfg.pipeline
    assert (
        cfg.pipeline["prototype_step"]["name"]
        == "fairlearn.preprocessing.PrototypeRepresentationLearner"
    )
    assert cfg.pipeline["prototype_step"]["max_iter"] == 1


def test_apply_defense_resolves_sensitive_ids_for_typed_pipeline():
    cfg = _cfg()
    cfg._X = pd.DataFrame(
        {
            "age": [25, 42],
            "hours_per_week": [40, 50],
            "sex": ["Male", "Female"],
        },
    )
    cfg.sensitive_columns = ["sex"]
    cfg.fairness_defense = {"name": "fairlearn.preprocessing.CorrelationRemover"}
    cfg.pipeline = {
        "imputer": {
            "name": "sklearn.impute.SimpleImputer",
            "strategy": "mean",
            "dtype": "numeric",
        },
    }

    cfg.apply_defense()

    fairness_step = cfg.pipeline["fairness_correlation_remover"]
    assert fairness_step["sensitive_feature_ids"] == [2]


def test_apply_defense_resolves_onehot_sensitive_ids_from_post_transform_columns():
    cfg = _cfg()
    cfg._X = pd.DataFrame(
        {
            "age": [25, 42, 33, 51],
            "marital.status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Never-married",
            ],
        },
    )
    cfg.sensitive_columns = ["marital.status"]
    cfg.fairness_defense = {"name": "fairlearn.preprocessing.CorrelationRemover"}
    cfg.pipeline = {
        "imputer": {
            "name": "sklearn.impute.SimpleImputer",
            "strategy": "mean",
            "dtype": "numeric",
        },
        "categorical_encoder": {
            "name": "sklearn.preprocessing.OneHotEncoder",
            "handle_unknown": "ignore",
            "sparse_output": False,
            "dtype": "object",
        },
    }

    cfg.apply_defense()

    fairness_step = cfg.pipeline["fairness_correlation_remover"]
    sensitive_ids = fairness_step["sensitive_feature_ids"]
    assert isinstance(sensitive_ids, list)
    assert len(sensitive_ids) >= 3


def test_load_data_validates_sensitive_columns(monkeypatch):
    cfg = _cfg()

    monkeypatch.setattr(DataConfig, "load_dataset", lambda self: self)
    cfg._X = pd.DataFrame({"group": ["a"], "x": [1]})
    cfg._y = pd.Series([0])
    cfg.sensitive_columns = None

    with pytest.raises(ValueError, match="must be configured"):
        cfg.load_dataset()


def test_sample_populates_sensitive_val_when_present(monkeypatch):
    cfg = _cfg()
    cfg.sensitive_columns = ["group"]

    def _noop_fit(self, run_hooks: bool = True):
        _ = run_hooks
        self._X = pd.DataFrame({"group": ["a", "b", "b", "a"], "x": [1, 2, 3, 4]})
        self.train_indices = [0, 1]
        self.test_indices = [2]
        self.val_indices = [3]
        self.X_train = self._X.iloc[self.train_indices].reset_index(drop=True)
        self.X_test = self._X.iloc[self.test_indices].reset_index(drop=True)
        self.X_val = self._X.iloc[self.val_indices].reset_index(drop=True)
        self.y_train = pd.Series([0, 1])
        self.y_test = pd.Series([1])
        self.y_val = pd.Series([0])

    monkeypatch.setattr(DataConfig, "fit", _noop_fit)

    cfg.fit()

    assert cfg._sensitive_train is not None
    assert cfg._sensitive_test is not None
    assert cfg._sensitive_all is not None
    assert cfg._sensitive_val is not None
    assert cfg._sensitive_train.tolist() == ["a", "b"]
    assert cfg._sensitive_test.tolist() == ["b"]
    assert cfg._sensitive_val.tolist() == ["a"]


def test_sample_prefers_post_transform_sensitive_columns(monkeypatch):
    cfg = _cfg()
    cfg.sensitive_columns = ["marital.status"]

    def _noop_fit(self, run_hooks: bool = True):
        _ = run_hooks
        self._X = pd.DataFrame(
            {
                "marital.status": [
                    "Never-married",
                    "Married-civ-spouse",
                    "Divorced",
                ],
                "age": [21, 47, 36],
            },
        )
        self.train_indices = [0, 1]
        self.test_indices = [2]
        self.val_indices = None
        self.X_train = pd.DataFrame(
            {
                "marital.status_Never-married": [1, 0],
                "marital.status_Married-civ-spouse": [0, 1],
                "marital.status_Divorced": [0, 0],
                "age": [21, 47],
            },
        )
        self.X_test = pd.DataFrame(
            {
                "marital.status_Never-married": [0],
                "marital.status_Married-civ-spouse": [0],
                "marital.status_Divorced": [1],
                "age": [36],
            },
        )
        self.y_train = pd.Series([0, 1])
        self.y_test = pd.Series([1])
        self.y_val = None

    monkeypatch.setattr(DataConfig, "fit", _noop_fit)

    cfg.fit()

    assert isinstance(cfg._sensitive_train, pd.DataFrame)
    assert isinstance(cfg._sensitive_test, pd.DataFrame)
    assert cfg._sensitive_train.shape[1] >= 3
    assert cfg._sensitive_test.shape[1] >= 3


def test_score_none_and_non_callable_paths():
    cfg = _cfg()
    cfg._y = pd.Series([0, 1])
    cfg._X = pd.DataFrame({"x": [1, 2]})
    cfg.scorer = None
    assert cfg.score() == {}

    cfg.scorer = "not-callable"
    with pytest.raises(TypeError, match="must be callable"):
        cfg.score()


def test_call_delegates_to_canonical_runtime(monkeypatch):
    cfg = _cfg()
    seen = {}

    def _execute_data_runtime(self, *args, files=None, **kwargs):
        self.X_train = pd.DataFrame({"x": [1]})
        seen["self"] = self
        seen["args"] = args
        seen["files"] = files
        seen["kwargs"] = kwargs
        return {"runtime": "ok"}

    monkeypatch.setattr(DataConfig, "execute_data_runtime", _execute_data_runtime)

    result = cfg("payload", files={"score_file": "scores.json"}, mode="test")

    assert result == {"runtime": "ok"}
    assert seen["self"] is cfg
    assert seen["args"] == ("payload",)
    assert seen["files"] == {"score_file": "scores.json"}
    assert seen["kwargs"] == {"mode": "test"}
