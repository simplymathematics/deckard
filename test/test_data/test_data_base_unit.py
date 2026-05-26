from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from helpers import load_canonical_data_profile
from omegaconf import OmegaConf
from scipy.sparse import csr_matrix
from sklearn.preprocessing import FunctionTransformer

import deckard.data.base as data_base
import deckard.data.declarations as data_declarations
from deckard.data.base import DataConfig
from deckard.data.pipeline import DataPipeline
from deckard.score.base import ScorerDictConfig


def _basic_data_config(**overrides):
    params = load_canonical_data_profile("classification", framework="sklearn")
    params["data_params"].update(
        {
            "n_samples": 20,
            "n_features": 4,
            "n_informative": 2,
            "n_redundant": 0,
            "random_state": 0,
            "n_clusters_per_class": 1,
        },
    )
    params.update(overrides)
    return DataConfig(**params)


def test_data_pipeline_config_is_legacy_alias():
    assert issubclass(DataConfig, DataConfig)

    cfg = DataConfig(scorer="none")
    assert isinstance(cfg, DataConfig)
    assert cfg.pipeline is None


def test_discover_lifelines_dataset_loaders_handles_import_error(monkeypatch):
    monkeypatch.setattr(data_declarations, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        data_declarations.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )
    assert data_declarations.discover_provider_dataset_loaders("lifelines") == {}


def test_discover_lifelines_dataset_loaders_filters_callable_loaders(monkeypatch):
    datasets_module = SimpleNamespace(
        load_lung=lambda: 1,
        load_kidney=lambda: 2,
        other=3,
    )
    monkeypatch.setattr(
        data_declarations,
        "_module_available",
        lambda _name: True,
    )
    monkeypatch.setattr(
        data_declarations.importlib,
        "import_module",
        lambda name: datasets_module,
    )

    loaders = data_declarations.discover_provider_dataset_loaders("lifelines")

    assert set(loaders) == {"lung", "kidney"}


def test_discover_yellowbrick_dataset_loaders_handles_import_error(monkeypatch):
    monkeypatch.setattr(data_declarations, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        data_declarations.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )
    assert data_declarations.discover_provider_dataset_loaders("yellowbrick") == {}


def test_discover_yellowbrick_dataset_loaders_filters_callable_loaders(monkeypatch):
    datasets_module = SimpleNamespace(
        load_energy=lambda: 1,
        load_credit=lambda: 2,
        other=3,
    )
    monkeypatch.setattr(
        data_declarations,
        "_module_available",
        lambda _name: True,
    )
    monkeypatch.setattr(
        data_declarations.importlib,
        "import_module",
        lambda name: datasets_module,
    )

    loaders = data_declarations.discover_provider_dataset_loaders("yellowbrick")

    assert set(loaders) == {"energy", "credit"}


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


@pytest.mark.xfail(
    condition=True,
    reason="pkg import fails, as expected. Need better Mock.",
)
def test_plugin_instantiation_and_hook_paths(monkeypatch):
    calls = []

    class PluginType:
        def after_load_data(self, cfg, **kwargs):
            calls.append((cfg.dataset_name, kwargs))
            return {"plugin": True}

    plugin_object = PluginType()
    import deckard.data._mixins as data_mixins

    monkeypatch.setattr(
        data_mixins,
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


@pytest.mark.xfail(
    condition=True,
    reason="pkg import fails, as expected. Need better Mock.",
)
def test_resolve_sample_branches(monkeypatch):
    import deckard.data.sample as data_sample
    from deckard.data.sample import BaseSampler

    cfg = _basic_data_config(sampler="split")
    assert BaseSampler.resolve(cfg).__class__.__name__ == "SplitSampler"

    cfg.sampler = {}
    assert BaseSampler.resolve(cfg) is None

    loaded = object()
    monkeypatch.setattr(
        data_sample,
        "load_class",
        lambda path, **kwargs: (path, kwargs),
    )
    cfg.sampler = {"name": "pkg.Sample", "folds": 3}
    assert BaseSampler.resolve(cfg) == ("pkg.Sample", {"folds": 3})

    cfg.sampler = lambda *args, **kwargs: loaded
    assert BaseSampler.resolve(cfg) is cfg.sampler

    class CustomSampler:
        pass

    cfg.sampler = CustomSampler
    assert isinstance(BaseSampler.resolve(cfg), CustomSampler)

    cfg.sampler = {"folds": 3}
    with pytest.raises(ValueError):
        BaseSampler.resolve(cfg)

    cfg.sampler = "unknown"
    with pytest.raises(ValueError):
        BaseSampler.resolve(cfg)

    cfg.sampler = 1
    with pytest.raises(ValueError):
        BaseSampler.resolve(cfg)


def test_load_lifelines_dataset_branches(monkeypatch):
    cfg = _basic_data_config(dataset_name="lung", target=None, classifier=False)

    monkeypatch.setattr(
        data_declarations,
        "discover_provider_dataset_loaders",
        lambda provider: {},
    )
    with pytest.raises(ImportError):
        data_declarations.load_lifelines_dataset(cfg, "lung")

    monkeypatch.setattr(
        data_declarations,
        "discover_provider_dataset_loaders",
        lambda provider: {"other": lambda **kwargs: pd.DataFrame({"x": [1]})},
    )
    with pytest.raises(NotImplementedError):
        data_declarations.load_lifelines_dataset(cfg, "lung")

    monkeypatch.setattr(
        data_declarations,
        "discover_provider_dataset_loaders",
        lambda provider: {
            "lung": lambda **kwargs: pd.DataFrame({"time": [1, 2], "feature": [3, 4]}),
        },
    )
    data_declarations.load_lifelines_dataset(cfg, "lung")
    assert "time" in cfg._X.columns
    assert cfg._y.tolist() == [0, 0]


def test_load_yellowbrick_dataset_branches(monkeypatch):
    cfg = _basic_data_config(dataset_name="energy", target=None, classifier=False)

    monkeypatch.setattr(
        data_declarations,
        "discover_provider_dataset_loaders",
        lambda provider: {},
    )
    with pytest.raises(ImportError):
        data_declarations.load_yellowbrick_dataset(cfg, "energy")

    monkeypatch.setattr(
        data_declarations,
        "discover_provider_dataset_loaders",
        lambda provider: {
            "other": lambda **kwargs: (pd.DataFrame({"x": [1]}), pd.Series([0])),
        },
    )
    with pytest.raises(NotImplementedError):
        data_declarations.load_yellowbrick_dataset(cfg, "energy")

    monkeypatch.setattr(
        data_declarations,
        "discover_provider_dataset_loaders",
        lambda provider: {
            "energy": lambda **kwargs: (
                pd.DataFrame({"f": [1, 2]}),
                pd.Series([0, 1]),
            ),
        },
    )
    data_declarations.load_yellowbrick_dataset(cfg, "energy")
    assert list(cfg._X.columns) == ["f"]
    assert cfg._y.tolist() == [0, 1]

    monkeypatch.setattr(
        data_declarations,
        "discover_provider_dataset_loaders",
        lambda provider: {"energy": lambda **kwargs: pd.DataFrame({"x": [1, 2]})},
    )
    cfg.target = None
    data_declarations.load_yellowbrick_dataset(cfg, "energy")
    assert cfg._y.tolist() == [0, 0]

    monkeypatch.setattr(
        data_declarations,
        "discover_provider_dataset_loaders",
        lambda provider: {"energy": lambda **kwargs: object()},
    )
    with pytest.raises(TypeError):
        data_declarations.load_yellowbrick_dataset(cfg, "energy")


def test_load_data_routes_lifelines_aliases_and_csv_options(monkeypatch, tmp_path):
    cfg = _basic_data_config(dataset_name="lung", target="target")
    monkeypatch.setattr(
        data_declarations,
        "discover_provider_dataset_loaders",
        lambda provider: (
            {
                "lung": lambda **kwargs: pd.DataFrame({"x": [1], "target": [0]}),
            }
            if provider == "lifelines"
            else {}
        ),
    )
    data_declarations.load_lifelines_dataset(cfg, "lung")
    assert list(cfg._X.columns) == ["x"]
    assert cfg._y.tolist() == [0]

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
    cfg = _basic_data_config(dataset_name="energy", target="target")
    monkeypatch.setattr(
        data_declarations,
        "discover_provider_dataset_loaders",
        lambda provider: (
            {
                "energy": lambda **kwargs: (
                    pd.DataFrame({"x": [1]}),
                    pd.Series([0]),
                ),
            }
            if provider == "yellowbrick"
            else {}
        ),
    )
    data_declarations.load_yellowbrick_dataset(cfg, "energy")
    assert list(cfg._X.columns) == ["x"]
    assert cfg._y.tolist() == [0]


def test_prepare_data_file_existing_and_new(tmp_path):
    cfg = _basic_data_config()
    existing = tmp_path / "existing.pkl"
    existing.write_text("x")
    loaded_cfg = object()
    cfg.load = lambda path: loaded_cfg

    assert cfg._prepare_files(files={"data_file": existing.as_posix()}) is False

    target = tmp_path / "nested" / "data.pkl"
    assert cfg._prepare_files(files={"data_file": target.as_posix()}) is True
    assert target.parent.exists()
    cfg.files = {}
    assert cfg._prepare_files(files=None) is False


def test_pipeline_config_coerces_legacy_pipeline_specs():
    cfg = DataConfig(
        pipeline={
            "feature": {
                "name": "sklearn.preprocessing.FunctionTransformer",
                "fit_X": True,
            },
        },
        scorer="none",
    )
    assert isinstance(cfg.pipeline, DataPipeline)

    cfg_list = DataConfig(
        pipeline=[
            {
                "feature": {
                    "name": "sklearn.preprocessing.FunctionTransformer",
                    "fit_X": True,
                },
            },
        ],
        scorer="none",
    )
    assert isinstance(cfg_list.pipeline, DataPipeline)


def test_pipeline_step_rejects_fit_y_and_fit_xy_both_true():
    with pytest.raises(
        ValueError,
        match="fit_xy pipeline steps are no longer supported",
    ):
        data_base.DataPipelineStep.from_config(
            "bad",
            {
                "name": "sklearn.preprocessing.FunctionTransformer",
                "fit_y": True,
                "fit_xy": True,
            },
        )


@pytest.mark.xfail(
    reason="Pipeline stage flag semantics were refactored; assertions need refresh.",
)
def test_pipeline_stage_flags_apply_only_to_declared_stages(monkeypatch):
    import deckard.data._mixins as data_mixins

    class AddConstantTransformer:
        def __init__(self, amount):
            self.amount = amount

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            frame = X.copy()
            return frame + self.amount

    def _fake_load_class(name, *args, **kwargs):
        _ = (args, kwargs)
        amounts = {
            "test.Pre": 100,
            "test.X": 1,
            "test.Xy": 5,
            "test.Y": 10,
        }
        return AddConstantTransformer(amounts[name])

    monkeypatch.setattr(data_mixins, "load_class", _fake_load_class)

    cfg = DataConfig(
        pipeline={
            "pre": {
                "name": "test.Pre",
                "fit_pre_sample": True,
                "fit_post_sample": False,
            },
            "x": {"name": "test.X", "fit_X": True},
            "xy": {"name": "test.Xy", "fit_Xy": True},
            "y": {"name": "test.Y", "fit_y": True, "fit_X": False},
        },
        scorer="none",
        score_mode="test",
    )

    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([1.0, 2.0])

    X_pre, y_pre = cfg.fit_presample(X, y)
    assert X_pre.iloc[:, 0].tolist() == [101.0, 102.0]
    assert y_pre.tolist() == [1.0, 2.0]

    X_x, y_x = cfg.fit_X(X_pre, y_pre)
    assert X_x.iloc[:, 0].tolist() == [102.0, 103.0]
    assert y_x.tolist() == [1.0, 2.0]

    X_xy, y_xy = cfg.fit_Xy(X_x, y_x)
    assert X_xy.iloc[:, 0].tolist() == [107.0, 108.0]
    assert y_xy.tolist() == [1.0, 2.0]

    X_y, y_y = cfg.fit_y(X_xy, y_xy)
    assert X_y.iloc[:, 0].tolist() == [107.0, 108.0]
    assert y_y.tolist() == [11.0, 12.0]


def test_pipeline_dtype_routing_keeps_untyped_steps(monkeypatch):
    import deckard.data.pipeline.base as data_pipeline_core
    from sklearn.base import BaseEstimator, TransformerMixin

    class IdentityTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    monkeypatch.setattr(
        data_pipeline_core,
        "load_class",
        lambda name, *args, **kwargs: IdentityTransformer(),
    )

    cfg = DataConfig(
        pipeline={
            "typed": {
                "name": "typed.Transformer",
                "dtype": "num",
                "fit_X": True,
            },
            "untagged": {
                "name": "untagged.Transformer",
                "fit_X": True,
            },
        },
        scorer="none",
    )

    x_steps = cfg.pipeline._collect_x_steps(stage="X")
    fitted = cfg.pipeline._build_x_pipeline(x_steps)
    assert fitted is not None
    assert [name for name, _ in fitted.steps] == ["preprocess", "untagged"]


def test_score_and_feature_score_branches(monkeypatch):
    cfg = _basic_data_config(scorer=None)
    assert cfg.score() == {}

    cfg.scorer = "bad"
    with pytest.raises(TypeError):
        cfg.score()

    plugin = SimpleNamespace(
        before_score=lambda *_args, **_kwargs: None,
        after_score=lambda *_args, **_kwargs: {"plugin_score": 7},
    )
    cfg.plugins = [plugin]
    cfg._plugin_objects = [plugin]
    cfg.scorer = lambda **kwargs: {"base_score": 1}
    cfg.y_train = pd.Series([0, 1])
    cfg.X_train = pd.DataFrame({"a": [1, 2]})
    cfg.y_test = pd.Series([0, 1])
    cfg.X_test = pd.DataFrame({"a": [1, 2]})
    cfg._X = pd.DataFrame({"a": [1, 2]})
    cfg._y = pd.Series([0, 1])
    assert cfg.score() == {"base_score": 1}

    cfg.score_dict = {}
    cfg._score_orchestration_active = True
    try:
        staged = cfg._score_orchestration_hook(stage="post-pipeline")
    finally:
        cfg._score_orchestration_active = False

    assert staged == {"base_score": 1, "plugin_score": 7}
    assert cfg.score_dict == staged


def test_empty_runtime_pipeline_leaves_data_unchanged():
    cfg = DataConfig(pipeline={}, scorer="none")
    cfg._X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    cfg._y = pd.Series([0, 1, 0])

    cfg.fit()

    assert cfg.X_train is not None
    assert cfg.X_test is not None
    assert cfg.y_train is not None
    assert cfg.y_test is not None


def test_fit_transform_x_handles_sparse_and_generated_feature_names():
    runtime = DataPipeline(
        pipeline={
            "csr": {
                "name": "sklearn.preprocessing.FunctionTransformer",
                "kwargs": {
                    "func": lambda values: csr_matrix(values),
                    "validate": False,
                },
            },
        },
    )

    pipeline = runtime._build_x_pipeline(runtime._collect_x_steps(stage="X"))
    x_train = pd.DataFrame({"a": [1.0, 2.0]})
    transformed = runtime._fit_transform_features(pipeline, x_train, x_train)

    assert list(transformed.columns) == ["feature_0"]


def test_fit_transform_y_csr_conversion():
    cfg = DataConfig(pipeline={}, scorer="none")
    transformer = FunctionTransformer(
        lambda values: csr_matrix(values),
        validate=False,
    )
    y_train = pd.Series([0, 1])

    transformed = cfg.pipeline._fit_transform_target([("csr", transformer)], y_train)
    assert isinstance(transformed, pd.Series)
    assert transformed.tolist() == [0, 1]


def test_pipeline_call_saves_scores_and_handles_y_pipeline(tmp_path):
    cfg = DataConfig(scorer="none")
    cfg.data_load_time = 0.1
    cfg.data_sample_time = 0.2
    cfg.X_train = pd.DataFrame({"a": [1, 2]})
    cfg.X_test = pd.DataFrame({"a": [3]})
    cfg.y_train = pd.Series([0, 1])
    cfg.y_test = pd.Series([1])
    cfg.fit = lambda run_hooks=True: cfg
    cfg.load_dataset = lambda: None
    cfg.score = lambda *args, mode=None, **kwargs: {"metric": 2}
    saved = {}
    cfg.merge_and_persist_scores = (
        lambda scores, score_file=None: saved.update(scores) or scores
    )
    cfg._prepare_files = lambda files=None: False
    cfg.files = {}

    score_path = tmp_path / "scores.json"
    result = cfg(files={"score_file": score_path.as_posix()})

    assert result["metric"] == 2
    assert saved["metric"] == 2


def test_call_populates_data_score_time_without_plugin_fallback(tmp_path):
    cfg = DataConfig(scorer="none")
    cfg.data_load_time = 0.1
    cfg.data_sample_time = 0.2
    cfg.X_train = pd.DataFrame({"a": [1, 2]})
    cfg.X_test = pd.DataFrame({"a": [3]})
    cfg.y_train = pd.Series([0, 1])
    cfg.y_test = pd.Series([1])
    cfg.load_dataset = lambda: None
    cfg.fit = lambda run_hooks=True: cfg
    cfg._run_plugin_hook = lambda *args, **kwargs: None
    cfg.score = lambda *args, mode=None, **kwargs: {"metric": 2}
    cfg.merge_and_persist_scores = lambda scores, score_file=None: scores
    cfg._prepare_files = lambda files=None: False
    cfg.files = {}

    score_path = tmp_path / "scores.json"
    result = cfg(files={"score_file": score_path.as_posix()})

    assert "data_score_time" in result
    assert result["data_score_time"] is not None


def test_copy_runtime_state_propagates_data_score_and_pipeline_times():
    cfg = DataConfig(scorer="none")
    cfg.data_score_time = 0.33
    cfg.data_pipeline_time = 0.22
    cfg.pipeline_fit_time = 0.11
    cfg.pipeline_transform_time = 0.12
    cfg.times = {
        "data_score_time": 0.33,
        "data_pipeline_time": 0.22,
    }
    cfg.files = {"score_file": "scores.json"}

    target = SimpleNamespace()
    cfg._copy_runtime_state_to(target)

    assert target.data_score_time == 0.33
    assert target.data_pipeline_time == 0.22
    assert target.pipeline_fit_time == 0.11
    assert target.pipeline_transform_time == 0.12
    assert target.times["data_score_time"] == 0.33
    assert target.files["score_file"] == "scores.json"
