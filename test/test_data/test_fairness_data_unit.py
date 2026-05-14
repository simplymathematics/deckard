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

    monkeypatch.setattr(DataPipelineConfig, "_load_data", lambda self: None)
    cfg._X = pd.DataFrame({"group": ["a"], "x": [1]})
    cfg._y = pd.Series([0])
    cfg.sensitive_columns = None

    with pytest.raises(ValueError, match="must be configured"):
        cfg._load_data()


def test_sample_populates_sensitive_val_when_present(monkeypatch):
    cfg = _cfg()
    cfg.sensitive_columns = ["group"]

    def _noop_sample(self, run_hooks: bool = True):
        self.X_train = pd.DataFrame({"group": ["a", "b"], "x": [1, 2]})
        self.X_test = pd.DataFrame({"group": ["b"], "x": [3]})
        self._X = pd.DataFrame({"group": ["a", "b", "b"], "x": [1, 2, 3]})
        self.X_val = pd.DataFrame({"group": ["a"], "x": [4]})

    monkeypatch.setattr(DataPipelineConfig, "_sample", _noop_sample)

    cfg._sample()

    assert cfg._sensitive_train is not None
    assert cfg._sensitive_test is not None
    assert cfg._sensitive_all is not None
    assert cfg._sensitive_val is not None


def test_score_none_and_non_callable_paths():
    cfg = _cfg()
    cfg._y = pd.Series([0, 1])
    cfg._X = pd.DataFrame({"x": [1, 2]})
    cfg.scorer = None
    assert cfg._score() == {}

    cfg.scorer = "not-callable"
    with pytest.raises(TypeError, match="must be callable"):
        cfg._score()
