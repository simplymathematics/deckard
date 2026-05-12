from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.preprocessing import FunctionTransformer

import deckard.data.base as data_base
from deckard.data.base import DataConfig, DataPipelineConfig
from deckard.score.base import ScorerDictConfig


def _basic_data_config(**overrides):
    params = {
        "dataset_name": "make_classification",
        "data_params": {
            "n_samples": 20,
            "n_features": 4,
            "n_informative": 2,
            "n_redundant": 0,
            "random_state": 0,
            "n_clusters_per_class": 1,
        },
        "test_size": 0.25,
        "random_state": 0,
        "classifier": True,
    }
    params.update(overrides)
    return DataConfig(**params)


def test_discover_lifelines_dataset_loaders_handles_import_error(monkeypatch):
    monkeypatch.setattr(
        data_base.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )
    assert data_base._discover_lifelines_dataset_loaders() == {}


def test_discover_lifelines_dataset_loaders_filters_callable_loaders(monkeypatch):
    datasets_module = SimpleNamespace(
        load_lung=lambda: 1,
        load_kidney=lambda: 2,
        other=3,
    )
    monkeypatch.setattr(
        data_base.importlib,
        "import_module",
        lambda name: datasets_module,
    )

    loaders = data_base._discover_lifelines_dataset_loaders()

    assert set(loaders) == {"lung", "kidney"}


def test_discover_yellowbrick_dataset_loaders_handles_import_error(monkeypatch):
    monkeypatch.setattr(
        data_base.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )
    assert data_base._discover_yellowbrick_dataset_loaders() == {}


def test_discover_yellowbrick_dataset_loaders_filters_callable_loaders(monkeypatch):
    datasets_module = SimpleNamespace(
        load_energy=lambda: 1,
        load_credit=lambda: 2,
        other=3,
    )
    monkeypatch.setattr(
        data_base.importlib,
        "import_module",
        lambda name: datasets_module,
    )

    loaders = data_base._discover_yellowbrick_dataset_loaders()

    assert set(loaders) == {"energy", "credit"}


@pytest.mark.parametrize("test_size", [1.2, "bad"])
def test_validate_init_rejects_invalid_test_size(test_size):
    with pytest.raises(ValueError):
        _basic_data_config(test_size=test_size)


def test_resolve_max_samples_and_apply_max_samples_branches(monkeypatch):
    cfg = _basic_data_config()
    monkeypatch.setenv(data_base.DECKARD_TEST_MAX_SAMPLES_ENV, "-1")
    assert cfg._resolve_max_samples(5) is None

    monkeypatch.setenv(data_base.DECKARD_TEST_MAX_SAMPLES_ENV, "not-an-int")
    with pytest.raises(ValueError):
        cfg._resolve_max_samples(5)

    cfg._X = pd.Series([1, 2, 3, 4])
    cfg._y = pd.Series([0, 1, 0, 1])
    monkeypatch.setenv(data_base.DECKARD_TEST_MAX_SAMPLES_ENV, "2")
    cfg._apply_max_samples()
    assert len(cfg._X) == 2
    assert len(cfg._y) == 2

    cfg._X = np.array([[1], [2], [3]])
    cfg._y = pd.Series([0, 1, 0])
    with pytest.raises(TypeError):
        cfg._apply_max_samples()


def test_post_init_normalizes_scorer_string_and_dict(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        data_base,
        "load_class",
        lambda path, *args, **kwargs: sentinel,
    )

    none_cfg = _basic_data_config(scorer="none")
    assert none_cfg.scorer is None

    auto_cfg = _basic_data_config(classifier=False, scorer="default")
    assert auto_cfg.scorer is sentinel

    dict_cfg = _basic_data_config(scorer=OmegaConf.create({"scorers": {}}))
    assert isinstance(dict_cfg.scorer, ScorerDictConfig)


def test_plugin_instantiation_and_hook_paths(monkeypatch):
    calls = []

    class PluginType:
        def after_load_data(self, cfg, **kwargs):
            calls.append((cfg.dataset_name, kwargs))
            return {"plugin": True}

    plugin_object = PluginType()
    monkeypatch.setattr(
        data_base,
        "load_class",
        lambda path, **kwargs: PluginType() if path else None,
    )

    cfg = _basic_data_config(
        plugins=[
            {"name": "pkg.Plugin", "flag": True},
            "pkg.Plugin",
            PluginType,
            plugin_object,
        ],
    )

    plugins = cfg._get_plugins()
    assert len(plugins) == 4
    assert cfg._get_plugins() is plugins

    outputs = cfg._run_plugin_hook("after_load_data", source="unit")
    assert len(outputs) == 4
    assert calls

    with pytest.raises(ValueError):
        cfg._instantiate_plugin({"flag": True})

    cfg.plugins = "pkg.Plugin"
    cfg._plugin_objects = None
    with pytest.raises(TypeError):
        cfg._get_plugins()


def test_get_stratify_col_branches():
    cfg = _basic_data_config(stratify=True)
    cfg._X = pd.DataFrame({"group": ["a", "b"], "x": [1, 2]})
    cfg._y = pd.Series([0, 1])
    assert cfg._get_stratify_col().equals(cfg._y)

    cfg.classifier = False
    assert cfg._get_stratify_col() is None

    cfg.classifier = True
    cfg.stratify = "group"
    assert cfg._get_stratify_col().tolist() == ["a", "b"]

    cfg.stratify = "missing"
    with pytest.raises(ValueError):
        cfg._get_stratify_col()

    cfg.stratify = 3
    with pytest.raises(ValueError):
        cfg._get_stratify_col()


def test_resolve_sample_branches(monkeypatch):
    cfg = _basic_data_config(sample="split")
    assert cfg._resolve_sample().__class__.__name__ == "SplitSampler"

    cfg.sample = {}
    assert cfg._resolve_sample() is None

    loaded = object()
    monkeypatch.setattr(data_base, "load_class", lambda path, **kwargs: (path, kwargs))
    cfg.sample = {"name": "pkg.Sample", "folds": 3}
    assert cfg._resolve_sample() == ("pkg.Sample", {"folds": 3})

    cfg.sample = lambda *args, **kwargs: loaded
    assert cfg._resolve_sample() is cfg.sample

    class CustomSampler:
        pass

    cfg.sample = CustomSampler
    assert isinstance(cfg._resolve_sample(), CustomSampler)

    cfg.sample = {"folds": 3}
    with pytest.raises(ValueError):
        cfg._resolve_sample()

    cfg.sample = "unknown"
    with pytest.raises(ValueError):
        cfg._resolve_sample()

    cfg.sample = 1
    with pytest.raises(ValueError):
        cfg._resolve_sample()


def test_load_lifelines_dataset_branches(monkeypatch):
    cfg = _basic_data_config(dataset_name="lung", target=None, classifier=False)

    monkeypatch.setattr(data_base, "_lifelines_dataset_loaders", lambda: {})
    with pytest.raises(ImportError):
        cfg._load_lifelines_dataset("lung")

    monkeypatch.setattr(
        data_base,
        "_lifelines_dataset_loaders",
        lambda: {"other": lambda: pd.DataFrame({"x": [1]})},
    )
    with pytest.raises(NotImplementedError):
        cfg._load_lifelines_dataset("lung")

    monkeypatch.setattr(
        data_base,
        "_lifelines_dataset_loaders",
        lambda: {
            "lung": lambda **kwargs: pd.DataFrame({"time": [1, 2], "feature": [3, 4]}),
        },
    )
    cfg._load_lifelines_dataset("lung")
    assert "time" in cfg._X.columns
    assert cfg._y.tolist() == [0, 0]


def test_load_yellowbrick_dataset_branches(monkeypatch):
    cfg = _basic_data_config(dataset_name="energy", target=None, classifier=False)

    monkeypatch.setattr(data_base, "_yellowbrick_dataset_loaders", lambda: {})
    with pytest.raises(ImportError):
        cfg._load_yellowbrick_dataset("energy")

    monkeypatch.setattr(
        data_base,
        "_yellowbrick_dataset_loaders",
        lambda: {"other": lambda: (pd.DataFrame({"x": [1]}), pd.Series([0]))},
    )
    with pytest.raises(NotImplementedError):
        cfg._load_yellowbrick_dataset("energy")

    monkeypatch.setattr(
        data_base,
        "_yellowbrick_dataset_loaders",
        lambda: {
            "energy": lambda **kwargs: (
                pd.DataFrame({"f": [1, 2]}),
                pd.Series([0, 1]),
            ),
        },
    )
    cfg._load_yellowbrick_dataset("energy")
    assert list(cfg._X.columns) == ["f"]
    assert cfg._y.tolist() == [0, 1]

    monkeypatch.setattr(
        data_base,
        "_yellowbrick_dataset_loaders",
        lambda: {"energy": lambda **kwargs: pd.DataFrame({"x": [1, 2]})},
    )
    cfg.target = None
    cfg._load_yellowbrick_dataset("energy")
    assert cfg._y.tolist() == [0, 0]

    monkeypatch.setattr(
        data_base,
        "_yellowbrick_dataset_loaders",
        lambda: {"energy": lambda **kwargs: object()},
    )
    with pytest.raises(TypeError):
        cfg._load_yellowbrick_dataset("energy")


def test_load_data_routes_lifelines_aliases_and_csv_options(monkeypatch, tmp_path):
    cfg = _basic_data_config(dataset_name="lifelines_demo", target="target")
    called = []
    monkeypatch.setattr(
        data_base,
        "_lifelines_dataset_loaders",
        lambda: {"demo": object()},
    )
    monkeypatch.setattr(
        cfg,
        "_load_lifelines_dataset",
        lambda name, **params: called.append(name)
        or setattr(cfg, "_X", pd.DataFrame({"x": [1]}))
        or setattr(cfg, "_y", pd.Series([0]))
        or setattr(cfg, "data_load_time", 0.01),
    )
    cfg._load_data()
    assert called == ["demo"]

    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"keep": [1, 2], "drop": [3, 4], "target": [0, 1]}).to_csv(
        csv_path,
        index=False,
    )

    cfg = _basic_data_config(
        dataset_name=csv_path.as_posix(),
        target="target",
        keep=["keep"],
    )
    cfg._load_from_csv()
    assert isinstance(cfg._X, pd.Series)
    assert cfg._X.name == "keep"

    cfg = _basic_data_config(
        dataset_name=csv_path.as_posix(),
        target="target",
        drop=["drop"],
    )
    cfg._load_from_csv()
    assert list(cfg._X.columns) == ["keep"]

    cfg = _basic_data_config(
        dataset_name=csv_path.as_posix(),
        target="target",
        keep=["keep"],
        drop=["drop"],
    )
    with pytest.raises(AssertionError):
        cfg._load_from_csv()


def test_load_data_routes_yellowbrick_aliases(monkeypatch):
    cfg = _basic_data_config(dataset_name="yellowbrick_energy", target="target")
    called = []
    monkeypatch.setattr(data_base, "_lifelines_dataset_loaders", lambda: {})
    monkeypatch.setattr(
        data_base,
        "_yellowbrick_dataset_loaders",
        lambda: {"energy": object()},
    )
    monkeypatch.setattr(
        cfg,
        "_load_yellowbrick_dataset",
        lambda name, **params: called.append(name)
        or setattr(cfg, "_X", pd.DataFrame({"x": [1]}))
        or setattr(cfg, "_y", pd.Series([0]))
        or setattr(cfg, "data_load_time", 0.01),
    )
    cfg._load_data()
    assert called == ["energy"]


def test_prepare_data_file_existing_and_new(tmp_path):
    cfg = _basic_data_config()
    existing = tmp_path / "existing.pkl"
    existing.write_text("x")
    loaded_cfg = object()
    cfg.load = lambda path: loaded_cfg

    assert cfg._prepare_data_file(existing.as_posix()) == (loaded_cfg, False)

    target = tmp_path / "nested" / "data.pkl"
    assert cfg._prepare_data_file(target.as_posix()) is True
    assert target.parent.exists()
    assert cfg._prepare_data_file(None) is False


def test_pipeline_config_invalid_and_fit_y_paths(monkeypatch):
    with pytest.raises(AssertionError):
        DataPipelineConfig(pipeline={"bad": []})

    cfg = DataPipelineConfig(
        pipeline={
            "target": {
                "name": "sklearn.preprocessing.FunctionTransformer",
                "fit_y": True,
            },
            "feature": {
                "name": "sklearn.preprocessing.FunctionTransformer",
                "dtype": "unknown",
            },
        },
        scorer="none",
    )
    x_pipeline, y_pipeline = cfg._init_pipeline()
    assert isinstance(x_pipeline, data_base.Pipeline)
    assert y_pipeline[0][0] == "target"

    cfg.pipeline = {
        "bad": {"name": "sklearn.preprocessing.FunctionTransformer", "fit_xy": True},
    }
    with pytest.raises(ValueError):
        cfg._init_pipeline()

    cfg.pipeline = "invalid"
    with pytest.raises(ValueError):
        cfg._init_pipeline()


def test_score_and_feature_score_branches(monkeypatch):
    cfg = _basic_data_config(scorer=None)
    assert cfg._score() == {}

    cfg.scorer = "bad"
    with pytest.raises(TypeError):
        cfg._score()

    plugin = SimpleNamespace(
        before_score=lambda *_args, **_kwargs: None,
        after_score=lambda *_args, **_kwargs: {"plugin_score": 7},
    )
    cfg.plugins = [plugin]
    cfg._plugin_objects = [plugin]
    cfg.scorer = lambda **kwargs: {"base_score": 1}
    cfg.y_train = pd.Series([0, 1])
    cfg.X_train = pd.DataFrame({"a": [1, 2]})
    cfg._X = pd.DataFrame({"a": [1, 2]})
    cfg._y = pd.Series([0, 1])
    assert cfg._score() == {"base_score": 1, "plugin_score": 7}

    cfg.X_train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    cfg.y_train = pd.Series([1, 1])
    assert cfg._classification_feature_scores()["class_counts"] == {1: 2}

    cfg.y_train = pd.Series([0, 1])
    monkeypatch.setattr(
        data_base,
        "mutual_info_classif",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("mi")),
    )
    monkeypatch.setattr(
        data_base,
        "f_classif",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("anova")),
    )
    scores = cfg._classification_feature_scores()
    assert scores["class_counts"] == {0: 1, 1: 1}

    reg_cfg = _basic_data_config(classifier=False)
    reg_cfg.X_train = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [0.5, 1.5, 2.5, 3.5, 4.5]},
    )
    reg_cfg.y_train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    reg_cfg.y_test = pd.Series([2.0, 4.0, 6.0])
    reg_scores = reg_cfg._regression_feature_scores()
    assert set(reg_scores) == {
        "mutual_info_regression",
        "f_regression",
        "r_regression",
        "y_train_cdf",
        "y_test_cdf",
    }


def test_fit_transform_x_empty_pipeline_short_circuits():
    cfg = DataPipelineConfig(pipeline={}, scorer="none")
    x_train = pd.DataFrame({"a": [1, 2]})
    x_test = pd.DataFrame({"a": [3]})
    y_train = pd.Series([0, 1])
    y_test = pd.Series([1])

    out = cfg._fit_transform_X(
        x_train,
        x_test,
        y_train,
        y_test,
        data_base.Pipeline(steps=[]),
    )
    assert out == (x_train, x_test, y_train, y_test)
    assert cfg.pipeline_fit_time == 0.0
    assert cfg.pipeline_transform_time == 0.0


def test_fit_transform_x_handles_sparse_and_generated_feature_names():
    cfg = DataPipelineConfig(pipeline={}, scorer="none")
    x_train = np.array([[1.0], [2.0]])
    x_test = np.array([[3.0]])
    y_train = pd.Series([0, 1])
    y_test = pd.Series([1])
    pipeline = data_base.Pipeline(
        steps=[
            (
                "csr",
                FunctionTransformer(
                    lambda values: data_base.csr_matrix(values),
                    validate=False,
                ),
            ),
        ],
    )

    out_train, out_test, _, _ = cfg._fit_transform_X(
        x_train,
        x_test,
        y_train,
        y_test,
        pipeline,
    )
    assert list(out_train.columns) == ["feature_0"]
    assert list(out_test.columns) == ["feature_0"]


def test_fit_transform_y_csr_conversion():
    cfg = DataPipelineConfig(pipeline={}, scorer="none")
    transformer = FunctionTransformer(
        lambda values: data_base.csr_matrix(values),
        validate=False,
    )
    x_train = pd.DataFrame({"a": [1, 2]})
    x_test = pd.DataFrame({"a": [3]})
    y_train = pd.Series([0, 1])
    y_test = pd.Series([1])

    with pytest.raises(ValueError, match="1-dimensional"):
        cfg._fit_transform_y(
            x_train,
            x_test,
            y_train,
            y_test,
            [("csr", transformer)],
        )


def test_pipeline_call_saves_scores_and_handles_y_pipeline(tmp_path):
    cfg = DataPipelineConfig(
        pipeline={
            "target": {
                "name": "sklearn.preprocessing.FunctionTransformer",
                "fit_y": True,
            },
        },
        scorer="none",
    )
    cfg.data_load_time = 0.1
    cfg.data_sample_time = 0.2
    cfg.X_train = pd.DataFrame({"a": [1, 2]})
    cfg.X_test = pd.DataFrame({"a": [3]})
    cfg.y_train = pd.Series([0, 1])
    cfg.y_test = pd.Series([1])
    cfg._sample = lambda: None
    cfg._load_data = lambda: None
    cfg.read_or_initialize_scores = lambda path: {"existing": 1}
    cfg._score = lambda: {"metric": 2}
    cfg._fit_transform_y = lambda X_train, X_test, y_train, y_test, pipeline: (
        setattr(cfg, "pipeline_y_fit_time", 0.01)
        or setattr(cfg, "pipeline_y_transform_time", 0.01)
        or X_train,
        X_test,
        y_train,
        y_test,
    )
    saved = {}
    cfg.save_scores = lambda scores, path: saved.update(scores)
    cfg._prepare_data_file = lambda data_file: False

    score_path = tmp_path / "scores.json"
    result = cfg(score_file=score_path.as_posix())

    assert result["existing"] == 1
    assert result["metric"] == 2
    assert result["pipeline_fit_n"] == 2
    assert result["pipeline_transform_n"] == 1
    assert saved["metric"] == 2
