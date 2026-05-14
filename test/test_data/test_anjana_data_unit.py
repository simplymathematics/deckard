import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from deckard.plugins.anjana.data import AnjanaDataConfig
from deckard.plugins.anjana import data as anjana_data_module
from deckard.data.base import DataPipelineConfig


def _bare_cfg():
    cfg = AnjanaDataConfig.__new__(AnjanaDataConfig)
    cfg.pipeline = {}
    cfg.anjana_defense = None
    cfg.fairness_defense = None
    cfg.identifiers = None
    cfg.quasi_identifiers = None
    cfg.sensitive_columns = None
    cfg.hierarchy_interval_sizes = None
    cfg.hierarchy_fill_value = "*"
    cfg.target = None
    cfg.hierarchies = None
    cfg.sensitive_attribute = None
    cfg.scorer = None
    return cfg


def test_post_init_normalizes_list_like_fields(monkeypatch):
    cfg = _bare_cfg()
    cfg.anjana_defense = OmegaConf.create(
        [
            {"name": "anjana.one"},
            {"k": 2},
        ],
    )
    cfg.fairness_defense = OmegaConf.create(
        [
            {"name": "fairlearn.preprocessing.CorrelationRemover"},
            {"alpha": 0.1},
        ],
    )
    cfg.identifiers = OmegaConf.create(["id"])
    cfg.quasi_identifiers = "zip"
    cfg.sensitive_columns = "group"
    cfg.hierarchy_interval_sizes = OmegaConf.create({"zip": [5]})

    monkeypatch.setattr(DataPipelineConfig, "__post_init__", lambda self: None)
    monkeypatch.setattr(AnjanaDataConfig, "_validate_init", lambda self: None)

    cfg.__post_init__()

    assert cfg.anjana_defense == {"name": "anjana.one", "k": 2}
    assert cfg.fairness_defense == {
        "name": "fairlearn.preprocessing.CorrelationRemover",
        "alpha": 0.1,
    }
    assert cfg.identifiers == ["id"]
    assert cfg.quasi_identifiers == ["zip"]
    assert cfg.sensitive_columns == ["group"]
    assert cfg.hierarchy_interval_sizes == {"zip": [5]}


def test_interval_helpers_cover_integer_float_and_nan_paths():
    cfg = _bare_cfg()

    assert cfg._format_interval_label(1, 3) == "[1, 3)"
    assert cfg._format_interval_label(1.5, 3.0) == "[1.5, 3.0)"

    all_nan = cfg._build_interval_hierarchy_level(pd.Series([None, np.nan]), 2)
    assert all_nan.tolist() == ["*", "*"]

    mixed = cfg._build_interval_hierarchy_level(pd.Series([1.0, np.nan, 3.2]), 2)
    assert mixed.tolist() == ["[0, 2)", "*", "[2, 4)"]


def test_generate_anjana_hierarchy_dict_validation_and_fallback_paths():
    cfg = _bare_cfg()
    frame = pd.DataFrame({"age": [21, 27], "zip": [101, 102]})

    with pytest.raises(TypeError, match="requires a pandas.DataFrame source"):
        cfg.generate_anjana_hierarchy_dict(frame=None)

    with pytest.raises(ValueError, match="quasi_identifiers must be provided"):
        cfg.generate_anjana_hierarchy_dict(frame=frame, quasi_identifiers=[])

    with pytest.raises(KeyError, match="missing"):
        cfg.generate_anjana_hierarchy_dict(frame=frame, quasi_identifiers=["missing"])

    single = cfg.generate_anjana_hierarchy_dict(
        frame=frame,
        quasi_identifiers="age",
        interval_sizes="not-a-dict",
        fill_value="MASK",
    )
    assert single["age"][1].tolist() == ["MASK", "MASK"]


def test_sensitive_helpers_and_target_resolution_paths():
    cfg = _bare_cfg()
    frame = pd.DataFrame({"group": ["a", "b"], "region": [1, 2]})

    assert cfg._resolve_anjana_target_column() == "__deckard_target__"
    cfg.target = "label"
    assert cfg._resolve_anjana_target_column() == "label"

    cfg.sensitive_columns = "group"
    assert cfg._sensitive_labels_from_frame(frame).tolist() == ["a", "b"]

    cfg.sensitive_columns = ["group", "region"]
    assert cfg._sensitive_labels_from_frame(frame).tolist() == [("a", "1"), ("b", "2")]

    cfg.sensitive_columns = None
    with pytest.raises(ValueError, match="must be configured"):
        cfg._sensitive_labels_from_frame(frame)

    with pytest.raises(ValueError, match="empty"):
        cfg._validate_sensitive_runtime([], "ctx")
    with pytest.raises(ValueError, match="all null"):
        cfg._validate_sensitive_runtime([None, np.nan], "ctx")
    with pytest.raises(ValueError, match="blank"):
        cfg._validate_sensitive_runtime([" ", ""], "ctx")


def test_apply_anjana_defense_branch_paths(monkeypatch):
    cfg = _bare_cfg()

    with pytest.raises(TypeError, match="requires tabular pandas DataFrame"):
        cfg._build_anjana_frame()

    cfg._X = pd.DataFrame({"feature": [1, 2, 3]}, index=[10, 11, 12])
    cfg._y = pd.Series([0, 1, 0], index=[10, 11, 12])
    cfg.quasi_identifiers = ["feature"]
    cfg.identifiers = ["id"]
    cfg.sensitive_attribute = "label"

    cfg.anjana_defense = True
    with pytest.raises(ValueError, match="ambiguous"):
        cfg._apply_anjana_defense()

    cfg.anjana_defense = 3
    with pytest.raises(TypeError, match="must be a dict"):
        cfg._apply_anjana_defense()

    cfg.anjana_defense = {}
    with pytest.raises(ValueError, match="include a 'name' or '_target_'"):
        cfg._apply_anjana_defense()

    cfg.anjana_defense = {"name": "anjana.fake"}
    monkeypatch.setattr(anjana_data_module, "resolve_class", lambda _: "not-callable")
    with pytest.raises(TypeError, match="not callable"):
        cfg._apply_anjana_defense()

    def _wrong_type(**kwargs):
        _ = kwargs
        return [1, 2, 3]

    monkeypatch.setattr(anjana_data_module, "resolve_class", lambda _: _wrong_type)
    with pytest.raises(TypeError, match="must return pandas.DataFrame"):
        cfg._apply_anjana_defense()

    seen = {}

    def _drop_target(**kwargs):
        seen.update(kwargs)
        return kwargs["data"].iloc[:2].drop(columns=["__deckard_target__"])

    cfg.target = None
    cfg.hierarchies = {"feature": {0: np.array([1, 2, 3])}}
    cfg.anjana_defense = {"_target_": "anjana.fake", "k": 2}
    monkeypatch.setattr(anjana_data_module, "resolve_class", lambda _: _drop_target)

    cfg._apply_anjana_defense()

    assert seen["ident"] == ["id"]
    assert seen["quasi_ident"] == ["feature"]
    assert seen["sens_att"] == "label"
    np.testing.assert_array_equal(
        seen["hierarchies"]["feature"][0],
        np.array([1, 2, 3]),
    )
    assert cfg._X.index.tolist() == [10, 11]
    assert cfg._y.index.tolist() == [10, 11]


def test_apply_anjana_defense_signature_filtering_and_defaults(monkeypatch):
    cfg = _bare_cfg()
    cfg._X = pd.DataFrame({"feature": [1, 2, 3]}, index=[10, 11, 12])
    cfg._y = pd.Series([0, 1, 0], index=[10, 11, 12])
    cfg.identifiers = None
    cfg.quasi_identifiers = ["feature"]
    cfg.sensitive_attribute = "label"
    cfg.target = None
    cfg.anjana_defense = {"name": "anjana.fake", "k": 2}

    seen = {}

    def _strict_k_anonymity_like(data, ident, quasi_ident, k, supp_level, hierarchies):
        seen.update(
            {
                "data": data,
                "ident": ident,
                "quasi_ident": quasi_ident,
                "k": k,
                "supp_level": supp_level,
                "hierarchies": hierarchies,
            },
        )
        return data.copy()

    monkeypatch.setattr(anjana_data_module, "resolve_class", lambda _: _strict_k_anonymity_like)

    cfg._apply_anjana_defense()

    assert seen["ident"] == []
    assert seen["quasi_ident"] == ["feature"]
    assert seen["k"] == 2
    assert seen["supp_level"] == 100
    assert "feature" in seen["hierarchies"]


def test_load_init_sample_and_score_paths(monkeypatch):
    cfg = _bare_cfg()

    calls = []
    monkeypatch.setattr(
        DataPipelineConfig,
        "_load_data",
        lambda self: (calls.append("load"), self)[1],
    )
    monkeypatch.setattr(
        AnjanaDataConfig,
        "_apply_anjana_defense",
        lambda self: calls.append("defense"),
    )
    assert cfg._load_data() is cfg
    assert calls == ["load", "defense"]

    monkeypatch.setattr(
        AnjanaDataConfig,
        "_inject_fairness_defense_step",
        lambda self: calls.append("inject"),
    )
    monkeypatch.setattr(DataPipelineConfig, "_init_pipeline", lambda self: "pipeline")
    assert cfg._init_pipeline() == "pipeline"

    def _sample_with_sensitive(self):
        self.X_train = pd.DataFrame({"group": ["a", "b"], "x": [1, 2]})
        self.X_test = pd.DataFrame({"group": ["b"], "x": [3]})
        self._X = pd.DataFrame({"group": ["a", "b", "b"], "x": [1, 2, 3]})

    monkeypatch.setattr(DataPipelineConfig, "_sample", _sample_with_sensitive)
    cfg.sensitive_columns = ["group"]
    cfg._sample()
    assert cfg._sensitive_train.tolist() == ["a", "b"]
    assert cfg._sensitive_test.tolist() == ["b"]
    assert cfg._sensitive_all.tolist() == ["a", "b", "b"]

    cfg_none = _bare_cfg()
    monkeypatch.setattr(DataPipelineConfig, "_sample", lambda self: None)
    cfg_none._sample()

    cfg_score = _bare_cfg()
    cfg_score.scorer = None
    assert cfg_score._score() == {}

    cfg_score.scorer = "default"
    cfg_score.y_train = pd.Series([0, 1])
    cfg_score.X_train = pd.DataFrame({"x": [1, 2]})
    monkeypatch.setattr(
        anjana_data_module,
        "load_class",
        lambda path: (lambda **kwargs: {"path": path, "n": len(kwargs["y_true"])}),
    )
    scored = cfg_score._score()
    assert scored == {
        "path": "deckard.plugins.anjana.score.DefaultAnjanaDataScorerConfig",
        "n": 2,
    }

    cfg_score.scorer = 5
    with pytest.raises(TypeError, match="must be callable or None"):
        cfg_score._score()

    cfg_fallback = _bare_cfg()
    cfg_fallback.scorer = lambda **kwargs: {
        "rows": len(kwargs["y_true"]),
        "cols": list(kwargs["y_pred"].columns),
    }
    cfg_fallback._y = pd.Series([1, 0])
    cfg_fallback._X = pd.DataFrame({"z": [3, 4]})
    assert cfg_fallback._score() == {"rows": 2, "cols": ["z"]}
