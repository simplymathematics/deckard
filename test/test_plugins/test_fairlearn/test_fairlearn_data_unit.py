import pandas as pd
import pytest
from omegaconf import ListConfig

from deckard.data.base import DataPipelineConfig
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

    monkeypatch.setattr(DataPipelineConfig, "__post_init__", lambda self: None)
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


def test_inject_fairness_defense_step_branch_paths():
    cfg = _cfg()

    cfg.fairness_defense = True
    with pytest.raises(ValueError, match="ambiguous"):
        cfg._inject_fairness_defense_step()

    cfg.fairness_defense = 7
    with pytest.raises(TypeError, match="must be a dict"):
        cfg._inject_fairness_defense_step()

    cfg.fairness_defense = {"name": "fairlearn.preprocessing.CorrelationRemover"}
    cfg._X = None
    cfg._inject_fairness_defense_step()

    cfg._X = pd.DataFrame({"group": ["a", "b"], "x": [1, 2]})
    cfg.sensitive_columns = None
    with pytest.raises(ValueError, match="must be configured"):
        cfg._inject_fairness_defense_step()

    cfg.sensitive_columns = ["missing"]
    with pytest.raises(RuntimeError, match="Sensitive features not found"):
        cfg._inject_fairness_defense_step()

    cfg.sensitive_columns = ["group"]
    cfg.fairness_defense = {}
    with pytest.raises(ValueError, match="include a 'name' key"):
        cfg._inject_fairness_defense_step()

    cfg.fairness_defense = {
        "name": "fairlearn.preprocessing.CorrelationRemover",
        "step_name": "fairness_step",
    }
    cfg.pipeline = {"existing": {"name": "noop"}}
    cfg._inject_fairness_defense_step()
    assert "fairness_step" in cfg.pipeline

    before = dict(cfg.pipeline)
    cfg._inject_fairness_defense_step()
    assert cfg.pipeline == before


def test_load_data_validates_sensitive_columns(monkeypatch):
    cfg = _cfg()

    monkeypatch.setattr(DataPipelineConfig, "load_dataset", lambda self: self)
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

    monkeypatch.setattr(DataPipelineConfig, "fit", _noop_fit)

    cfg.split_data()

    assert cfg._sensitive_train is not None
    assert cfg._sensitive_test is not None
    assert cfg._sensitive_all is not None
    assert cfg._sensitive_val is not None
    assert cfg._sensitive_train.tolist() == ["a", "b"]
    assert cfg._sensitive_test.tolist() == ["b"]
    assert cfg._sensitive_val.tolist() == ["a"]


def test_score_none_and_non_callable_paths():
    cfg = _cfg()
    cfg._y = pd.Series([0, 1])
    cfg._X = pd.DataFrame({"x": [1, 2]})
    cfg.scorer = None
    assert cfg.score() == {}

    cfg.scorer = "not-callable"
    with pytest.raises(TypeError, match="must be callable"):
        cfg.score()
