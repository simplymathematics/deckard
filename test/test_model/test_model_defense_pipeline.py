from types import SimpleNamespace
import builtins
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from deckard.model.base import ModelConfig
from deckard.model.defense.base import DefenseConfig, DefenseStep
from deckard.model.defense.postprocessor import PostprocessorDefenseConfig


class DummyDataConfig:
    def __init__(self, X_train, y_train, X_test, y_test):
        X_train = pd.DataFrame(X_train)
        y_train = pd.Series(y_train)
        X_test = pd.DataFrame(X_test)
        y_test = pd.Series(y_test)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


class TestDefenseConfig:
    def setup_method(self):
        self.data = DummyDataConfig(
            X_train=pd.DataFrame(np.random.rand(100, 10)),
            y_train=pd.Series(np.random.randint(0, 2, size=100)),
            X_test=pd.DataFrame(np.random.rand(20, 10)),
            y_test=pd.Series(np.random.randint(0, 2, size=20)),
        )

        self.defense_config = DefenseConfig(defenses=[])

    def test_defense_config_initialization(self):
        # Test default initialization
        assert self.defense_config.defenses == []
        assert self.defense_config.plugins == []
        assert self.defense_config.score_dict == {}
        assert self.defense_config._target_ == "deckard.model.defense.base.DefenseConfig"

    def test_apply_defense_without_model(self):
        # DefenseConfig is pipeline-only, so estimator is required.
        with pytest.raises(ValueError):
            self.defense_config.apply(
                estimator=cast(Any, None),
                data=cast(Any, self.data),
            )

    def test_apply_defense_with_invalid_defense_name(self):
        # Pipeline configs no longer expose single-defense name parsing.
        assert not hasattr(self.defense_config, "parse_defense_name")

    def test_call_is_not_runtime_owner(self):
        with pytest.raises(NotImplementedError):
            self.defense_config()

    def test_apply_to_trained_model(self):
        class _IdentityDefense:
            def __init__(self):
                self.name = "art.defences.postprocessor.HighConfidence"
                self.defense_application_time = None

            def apply_to(self, estimator, data):
                _ = data
                return {"estimator": estimator}

        model = ModelConfig(
            name="sklearn.ensemble.RandomForestClassifier",
            classifier=True,
            model_params={"n_estimators": 5, "random_state": 42},
        )
        model.train(self.data.X_train, self.data.y_train)

        pipeline = DefenseConfig(
            defenses=[DefenseStep.from_defense(_IdentityDefense())],
        )

        defended = pipeline.apply(
            estimator=model.get_model(),
            data=cast(Any, self.data),
        )
        assert defended is not None
        assert pipeline.defense_application_time is not None

    def test_hash_function(self):
        # Test the hash function for DefenseConfig
        hash_value = hash(self.defense_config)
        assert isinstance(hash_value, int)

    def test_supported_defense_types(self):
        # Test supported defense types
        supported_types = [
            "detector",
            "preprocessor",
            "postprocessor",
            "trainer",
            "regularizer",
            "transformer",
        ]
        assert "postprocessor" in supported_types
        assert "unsupported_type" not in supported_types

    def test_hash_stable_after_apply_for_defense_config(self):
        """DefenseConfig hash remains stable after runtime-only apply attrs are set."""
        original_hash = hash(self.defense_config)
        self.defense_config.defense_application_time = 1.23
        setattr(self.defense_config, "_defense_applied_at", 1234567890.5)
        setattr(
            self.defense_config,
            "_runtime_defense_state",
            {"applied": True},
        )
        if hasattr(self.defense_config, "score_dict") and isinstance(
            self.defense_config.score_dict,
            dict,
        ):
            self.defense_config.score_dict["runtime"] = 1

        assert original_hash == hash(
            self.defense_config,
        ), "Hash changed after defense apply-time runtime updates"


class TestDefensePipelineConfigListCoerce:
    """DefensePipelineConfig.coerce() with a list should chain all specs."""

    spec_a = {
        "_target_": "deckard.model.defense.postprocessor.PostprocessorDefenseConfig",
        "name": "art.defences.postprocessor.HighConfidence",
        "defense_params": {"cutoff": 0.25},
    }
    spec_b = {
        "_target_": "deckard.model.defense.postprocessor.PostprocessorDefenseConfig",
        "name": "art.defences.postprocessor.ClassLabels",
        "defense_params": {"apply_fit": False, "apply_predict": True},
    }

    def test_list_of_two_specs_produces_two_defenses(self):
        result = DefenseConfig.coerce([self.spec_a, self.spec_b])
        assert isinstance(result, DefenseConfig)
        assert len(result.defenses) == 2

    def test_list_of_one_spec_produces_one_defense(self):
        result = DefenseConfig.coerce([self.spec_a])
        assert isinstance(result, DefenseConfig)
        assert len(result.defenses) == 1

    def test_list_order_is_preserved(self):
        result = DefenseConfig.coerce([self.spec_a, self.spec_b])
        assert result is not None
        assert "HighConfidence" in result.defenses[0].defense_name
        assert "ClassLabels" in result.defenses[1].defense_name

    def test_empty_list_produces_empty_pipeline(self):
        result = DefenseConfig.coerce([])
        assert isinstance(result, DefenseConfig)
        assert len(result.defenses) == 0

    def test_none_still_returns_none(self):
        result = DefenseConfig.coerce(None)
        assert result is None

    def test_single_explicit_target_dict_wraps_in_one_element_list(self):
        result = DefenseConfig.coerce(dict(self.spec_a))
        assert result is not None
        assert isinstance(result, DefenseConfig)
        assert len(result.defenses) == 1

    def test_raw_dict_without_target_is_rejected(self):
        with pytest.raises(TypeError, match="Defense config must be"):
            DefenseConfig.coerce(
                {
                    "name": "art.defences.postprocessor.HighConfidence",
                },
            )


def test_defense_behavior_defaults_signature_and_apply_to_paths(monkeypatch):
    defense = PostprocessorDefenseConfig.__new__(PostprocessorDefenseConfig)
    defense.name = None
    defense.classifier = True
    defense.model_params = {}
    defense.probability = False
    defense.alias = ""
    defense.name = "art.defences.postprocessor.HighConfidence"
    setattr(defense, "defense_params", None)
    setattr(defense, "score_dict", None)
    setattr(defense, "_target_", None)

    defense.__post_init__()

    assert defense._model is None
    assert defense.score_dict == {}
    assert (
        defense._target_
        == "deckard.model.defense.postprocessor.PostprocessorDefenseConfig"
    )
    assert defense.defense_params == {}

    defense.defense_params = cast(
        Any,
        OmegaConf.create({"outer": {"inner": [1, {"value": 2}]}, "flat": [3, 4]}),
    )
    signature = defense._defense_signature()
    assert signature[0] == "art.defences.postprocessor.HighConfidence"
    assert isinstance(signature[1], tuple)

    with pytest.raises(ValueError, match="Model is not fitted yet"):
        defense.get_model()

    with pytest.raises(ValueError, match="estimator must be provided"):
        defense.apply_to(estimator=cast(Any, None), data=cast(Any, object()))

    defense._model_config = cast(Any, SimpleNamespace(_model=None))
    monkeypatch.setattr(
        PostprocessorDefenseConfig,
        "apply_defense",
        lambda self, data: {"data": data, "model": self._model},
    )
    estimator = object()
    result = defense.apply_to(
        estimator=cast(Any, estimator),
        data=cast(Any, "payload"),
    )
    assert result == {"data": "payload", "model": estimator}
    assert defense._model_config._model is estimator


def test_parse_defense_name_and_get_art_class_edge_paths(monkeypatch):
    defense = PostprocessorDefenseConfig(
        name="sklearn.linear_model.LogisticRegression",
        classifier=True,
    )
    defense_type, subtype, defense_class = defense.parse_defense_name()
    assert defense_type is None
    assert subtype is None
    assert defense_class is None

    defense.defense_name = "short.Class"
    with pytest.raises(ImportError, match="Could not parse defense type"):
        defense.parse_defense_name()

    defense.defense_name = "art.defences.postprocessor.Missing"
    monkeypatch.setattr(
        "deckard.model.defense.base.resolve_class",
        lambda name: (_ for _ in ()).throw(AttributeError(name)),
    )
    with pytest.raises(ImportError, match="Could not import defense class"):
        defense.parse_defense_name()

    defense = PostprocessorDefenseConfig(
        name="art.defences.postprocessor.HighConfidence",
        classifier=True,
    )
    with pytest.raises(ValueError, match="name must be set"):
        defense.get_art_class(
            cast(Any, SimpleNamespace(X_train=np.zeros((2, 3)), y_train=[0, 1])),
        )

    import deckard.model.defense.base as defend_module

    custom_art = type("CustomArt", (), {})
    monkeypatch.setitem(
        defend_module._get_art_symbols()["classifier_dict"],
        "LogisticRegression",
        custom_art,
    )
    defense.name = "sklearn.linear_model.LogisticRegression"
    art_class, init_params = defense.get_art_class(
        SimpleNamespace(X_train=np.zeros((2, 3)), y_train=[0, 1]),
    )
    assert art_class is custom_art
    assert init_params["input_shape"] == (3,)
    assert init_params["nb_classes"] == 2
    assert init_params["preprocessing"] is None


def test_extract_art_wrapper_context_ignores_untyped_cached_wrappers():
    defense = PostprocessorDefenseConfig(
        name="art.defences.postprocessor.HighConfidence",
        model_name="sklearn.linear_model.LogisticRegression",
        classifier=True,
    )

    class CachedLike:
        def __init__(self):
            self.model = "unexpected-inner"
            self.preprocessing_defences = ["fake-pre"]
            self.postprocessing_defences = ["fake-post"]

    cached_like = CachedLike()
    defense._model = cast(Any, cached_like)

    (
        base_estimator,
        _art_class,
        _art_params,
        existing_preprocessors,
        existing_postprocessors,
    ) = defense._extract_art_wrapper_context(
        art_class=object,
        init_params={"preprocessing": None},
    )

    # Non-ART cached objects should not be treated as wrappers.
    assert base_estimator is cached_like
    assert existing_preprocessors == []
    assert existing_postprocessors == []


def test_extract_art_wrapper_context_prefers_explicit_wrapper_state(monkeypatch):
    defense = PostprocessorDefenseConfig(
        name="art.defences.postprocessor.HighConfidence",
        model_name="sklearn.linear_model.LogisticRegression",
        classifier=True,
    )
    state_base = SimpleNamespace(name="base-estimator")
    wrapper = SimpleNamespace(
        model="legacy-inner",
        preprocessing_defences=["A"],
        postprocessing_defences=["B"],
        preprocessing=(0.0, 1.0),
        clip_values=(0.0, 1.0),
    )
    defense._model = cast(Any, wrapper)

    monkeypatch.setattr(
        "deckard.model.defense.base._is_art_wrapper_instance",
        lambda obj: obj is wrapper,
    )
    monkeypatch.setattr(
        "deckard.model.defense.base._get_wrapper_state",
        lambda obj: {"wrapped_by_deckard": True, "base_estimator": state_base},
    )

    (
        base_estimator,
        art_class,
        art_params,
        existing_preprocessors,
        existing_postprocessors,
    ) = defense._extract_art_wrapper_context(
        art_class=dict,
        init_params={"preprocessing": None},
    )

    assert base_estimator is state_base
    assert art_class is wrapper.__class__
    assert art_params["clip_values"] == (0.0, 1.0)
    assert art_params["preprocessing"] == (0.0, 1.0)
    assert existing_preprocessors == ["A"]
    assert existing_postprocessors == ["B"]


def test_get_art_class_torch_requires_typed_base_estimator(monkeypatch):
    defense = PostprocessorDefenseConfig(
        name="art.defences.postprocessor.HighConfidence",
        classifier=True,
    )
    defense.name = "torch.nn.Linear"
    defense._model = cast(Any, SimpleNamespace(model="not-a-torch-module"))

    monkeypatch.setattr(
        "deckard.model.defense.base._is_art_torch_wrapper",
        lambda _obj: True,
    )
    monkeypatch.setattr(
        "deckard.model.defense.base._is_torch_model_instance",
        lambda _obj: False,
    )

    with pytest.raises(TypeError, match="Torch defenses require a torch.nn.Module"):
        defense.get_art_class(
            cast(Any, SimpleNamespace(X_train=np.zeros((4, 3)), y_train=np.array([0, 1, 0, 1]))),
        )


def test_is_torch_model_instance_handles_runtime_import_error(monkeypatch):
    import deckard.model.defense.base as defend_module

    original_import = builtins.__import__

    def _failing_torch_import(name, *args, **kwargs):
        if name == "torch":
            raise RuntimeError("torch init failed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_torch_import)

    assert defend_module._is_torch_model_instance(object()) is False


def test_get_art_class_torch_runtime_import_error_wrapped(monkeypatch):
    defense = PostprocessorDefenseConfig(
        name="art.defences.postprocessor.HighConfidence",
        classifier=True,
    )
    defense.name = "torch.nn.Linear"
    defense._model = cast(Any, object())

    monkeypatch.setattr(
        "deckard.model.defense.base._is_torch_model_instance",
        lambda _obj: True,
    )

    original_import = builtins.__import__

    def _failing_torch_import(name, *args, **kwargs):
        if name == "torch":
            raise RuntimeError("torch init failed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_torch_import)

    with pytest.raises(
        ImportError,
        match="Torch model defenses require optional dependency deckard\\[torch\\]",
    ):
        defense.get_art_class(
            cast(Any, SimpleNamespace(X_train=np.zeros((2, 3)), y_train=np.array([0, 1]))),
        )


def test_pipeline_coerce_plugin_context_and_stage_helpers(monkeypatch):
    pipeline = DefenseConfig.__new__(DefenseConfig)
    pipeline.defenses = []
    pipeline.plugins = []
    setattr(pipeline, "score_dict", None)
    pipeline._target_ = None
    pipeline.__post_init__()
    assert pipeline.score_dict == {}
    assert pipeline._target_ == "deckard.model.defense.base.DefenseConfig"

    assert not DefenseConfig._is_pipeline_target(7)
    assert not DefenseConfig._looks_like_single_defense_spec({"defenses": []})
    assert not DefenseConfig._looks_like_single_defense_spec(
        {"_target_": "pkg.OtherDefense"},
    )

    class ApplyObj:
        def apply_to(self, estimator, data):
            return estimator

    coerced_apply = DefenseConfig.coerce(cast(Any, ApplyObj()))
    assert coerced_apply is not None
    assert len(coerced_apply.defenses) == 1
    assert (
        len(
            DefenseConfig.coerce(
                [
                    {
                        "_target_": "deckard.model.defense.base.DefenseConfig",
                        "name": "art.defences.postprocessor.HighConfidence",
                    },
                ],
            ).defenses,
        )
        == 1
    )
    assert (
        len(
            DefenseConfig.coerce(
                {
                    "_target_": "deckard.model.defense.base.DefenseConfig",
                    "defenses": [],
                },
            ).defenses,
        )
        == 0
    )
    with pytest.raises(TypeError, match="Defense config must be"):
        DefenseConfig.coerce(cast(Any, 3))

    resolved_plugins = []

    def _resolve_plugin(name):
        def _factory(**kwargs):
            resolved_plugins.append((name, kwargs))
            return {"name": name, **kwargs}

        return _factory

    monkeypatch.setattr("deckard.model.defense.base.resolve_class", _resolve_plugin)
    plugin_pipeline = DefenseConfig(defenses=[])
    assert plugin_pipeline._instantiate_plugin(
        {"name": "pkg.Plugin", "flag": True},
    ) == {"name": "pkg.Plugin", "flag": True}
    with pytest.raises(ValueError, match="must include 'name' or '_target_'"):
        plugin_pipeline._instantiate_plugin({"flag": True})
    assert plugin_pipeline._instantiate_plugin("pkg.Plugin") == {"name": "pkg.Plugin"}

    class LocalPlugin:
        def __init__(self):
            self.tag = "local"

    assert isinstance(plugin_pipeline._instantiate_plugin(LocalPlugin), LocalPlugin)
    marker = object()
    assert plugin_pipeline._instantiate_plugin(marker) is marker

    setattr(plugin_pipeline, "plugins", "bad")
    plugin_pipeline._plugin_objects = None
    with pytest.raises(TypeError, match="plugins must be a list"):
        plugin_pipeline._get_plugins()

    plugin_pipeline.plugins = [
        {"name": "pkg.Plugin", "alpha": 1},
        "pkg.Plugin",
        LocalPlugin,
        marker,
    ]
    plugin_pipeline._plugin_objects = None
    assert len(plugin_pipeline._get_plugins()) == 4

    plugin_pipeline._run_plugin_hook = lambda hook_name, **kwargs: [
        " tuned ",
        {"stage": "attack"},
        {"defense_stage": "train"},
        None,
    ]
    assert plugin_pipeline.resolve_stage() == "train"


def test_pipeline_single_defense_coercion_and_context_inheritance(monkeypatch):
    pipeline = DefenseConfig(defenses=[])

    class CustomDefense:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FairDefense:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _resolve(name):
        if name == "pkg.CustomDefense":
            return CustomDefense
        if name == "deckard.plugins.fairlearn.model.FairlearnDefenseConfig":
            return FairDefense
        raise RuntimeError(name)

    monkeypatch.setattr("deckard.model.defense.base.resolve_class", _resolve)

    coerced_target = pipeline._coerce_single_defense(
        {"_target_": "pkg.CustomDefense", "x": 1},
    )
    assert isinstance(coerced_target, DefenseStep)
    assert isinstance(coerced_target.defense, CustomDefense)
    assert coerced_target.defense.kwargs == {"x": 1}

    coerced_fair = pipeline._coerce_single_defense(
        {
            "_target_": "deckard.plugins.fairlearn.model.FairlearnDefenseConfig",
            "eps": 0.1,
        },
    )
    assert isinstance(coerced_fair, DefenseStep)
    assert isinstance(coerced_fair.defense, FairDefense)
    assert coerced_fair.defense.kwargs["eps"] == 0.1

    monkeypatch.setattr(
        "deckard.model.defense.base.resolve_class",
        lambda name: (_ for _ in ()).throw(RuntimeError(name)),
    )
    with pytest.raises(RuntimeError):
        pipeline._coerce_single_defense(
            {
                "_target_": "deckard.plugins.fairlearn.model.FairlearnDefenseConfig",
            },
        )
    with pytest.raises(TypeError, match="Unsupported defense specification"):
        pipeline._coerce_single_defense(3)

    defense = SimpleNamespace(
        name=None,
        classifier=None,
        model_params=None,
        probability=False,
    )
    estimator = SimpleNamespace(
        model=SimpleNamespace(
            _estimator_type="classifier",
            get_params=lambda: {"depth": 3},
            predict_proba=lambda x: x,
        ),
    )
    pipeline._inherit_model_context(defense, estimator)
    assert defense.name.endswith("SimpleNamespace")
    assert defense.classifier is True
    assert defense.model_params == {"depth": 3}
    assert defense.probability is True

    reg_defense = SimpleNamespace(
        name="",
        classifier=None,
        model_params={},
        probability=False,
    )
    reg_estimator = SimpleNamespace(
        _estimator_type="regressor",
        get_params=lambda: {"alpha": 1},
    )
    pipeline._inherit_model_context(reg_defense, reg_estimator)
    assert reg_defense.classifier is False
    assert reg_defense.model_params == {"alpha": 1}

    assert pipeline.normalize_defenses(None) == []
    with pytest.raises(TypeError, match="Defense specs must use an explicit _target_"):
        pipeline.normalize_defenses(
            {"name": "art.defences.postprocessor.HighConfidence"},
        )


def test_pipeline_apply_validation_and_elapsed_fallback(monkeypatch):
    pipeline = DefenseConfig(defenses=[])

    with pytest.raises(ValueError, match="estimator must be provided"):
        pipeline.apply(estimator=cast(Any, None), data=cast(Any, object()))

    estimator = object()
    assert pipeline.apply(estimator=cast(Any, estimator), data=cast(Any, object())) is estimator
    assert pipeline.apply_to(estimator=cast(Any, estimator), data=cast(Any, object())) is estimator
    assert pipeline.apply_defense(estimator=cast(Any, estimator), data=cast(Any, object())) is estimator

    bad_pipeline = DefenseConfig(defenses=[SimpleNamespace(apply_to=1)])
    with pytest.raises(TypeError, match="must implement apply_to"):
        bad_pipeline.apply(estimator=cast(Any, object()), data=cast(Any, object()))

    hook_calls = []

    class TimelessDefense:
        name = "custom.timeless"
        defense_application_time = None
        data = None
        name = None
        classifier = None
        model_params = None
        probability = False

        def apply_to(self, estimator, data):
            _ = data
            hook_calls.append("apply")
            return {"wrapped": estimator}

    timed_pipeline = DefenseConfig(defenses=[TimelessDefense()])
    monkeypatch.setattr(
        timed_pipeline,
        "_run_plugin_hook",
        lambda hook_name, **kwargs: hook_calls.append(hook_name) or [],
    )
    monkeypatch.setattr(
        "deckard.model.defense.base.time.perf_counter",
        iter([10.0, 12.5]).__next__,
    )

    result = timed_pipeline.apply(
        estimator=cast(Any, "model"),
        data=cast(Any, "payload"),
    )
    assert result == {"wrapped": "model"}
    assert timed_pipeline.defense_application_time == 2.5
    assert timed_pipeline.defenses[0].data == "payload"
    assert "before_apply_defense" in hook_calls
    assert "after_apply_defense" in hook_calls
