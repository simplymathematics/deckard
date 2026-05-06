from types import SimpleNamespace
import builtins

import numpy as np
import pandas as pd
import pytest

from deckard.utils import ConfigBase

from deckard.model.fairness import (
    FairlearnDefenseConfig,
    FairlearnModelConfig,
    FairlearnPytorchModelConfig,
    _FairnessBehaviorMixin,
)

pytest.importorskip("fairlearn")


class _DummyMixin(_FairnessBehaviorMixin):
    pass


def _dummy_data():
    return SimpleNamespace(
        X_train=pd.DataFrame({"x": [1, 2]}, index=[0, 1]),
        X_test=pd.DataFrame({"x": [3, 4]}, index=[10, 11]),
        _X=pd.DataFrame({"x": [5, 6]}, index=[20, 21]),
        y_train=pd.Series([0, 1]),
        _sensitive_train=pd.Series(["a", "b"]),
        _sensitive_test=pd.Series(["a", "b"]),
        _sensitive_all=pd.Series(["a", "b"]),
    )


def test_runtime_sensitive_source_and_split_resolution_errors():
    d = _DummyMixin()
    d.data = _dummy_data()

    assert list(d._resolve_runtime_sensitive_source("train")) == ["a", "b"]
    assert list(d._resolve_runtime_sensitive_source("test")) == ["a", "b"]
    assert list(d._resolve_runtime_sensitive_source("all")) == ["a", "b"]

    with pytest.raises(NotImplementedError):
        d._resolve_runtime_sensitive_source("val")
    with pytest.raises(ValueError):
        d._resolve_runtime_sensitive_source("bad")

    assert d._resolve_scoring_split("train") == "train"
    assert d._resolve_scoring_split("test") == "test"
    assert d._resolve_scoring_split("attack") == "test"
    assert d._resolve_scoring_split("all") == "all"
    with pytest.raises(NotImplementedError):
        d._resolve_scoring_split("val")
    with pytest.raises(ValueError):
        d._resolve_scoring_split("unknown")


def test_validate_sensitive_series_checks_empty_null_blank():
    d = _DummyMixin()

    assert d._validate_sensitive_series(None, "ctx") is None

    with pytest.raises(ValueError, match="empty"):
        d._validate_sensitive_series([], "ctx")
    with pytest.raises(ValueError, match="all null"):
        d._validate_sensitive_series([None, np.nan], "ctx")
    with pytest.raises(ValueError, match="blank"):
        d._validate_sensitive_series([" ", ""], "ctx")


def test_infer_and_resolve_sensitive_features_for_batch_paths(monkeypatch):
    d = _DummyMixin()
    d.data = _dummy_data()

    assert d._infer_split_from_batch(d.data.X_train) == "train"
    assert d._infer_split_from_batch(d.data.X_test.copy()) == "test"

    batch = pd.DataFrame({"x": [1, 2]})
    assert d._resolve_sensitive_features_for_batch(batch, split="train") is not None

    d.data._sensitive_train = pd.Series(["a"])  # length mismatch
    assert d._resolve_sensitive_features_for_batch(batch, split="train") is None

    d.data._sensitive_train = pd.Series(["a", "b"])
    monkeypatch.setattr(pd.Series, "reindex", lambda self, idx: (_ for _ in ()).throw(RuntimeError("reindex fail")))
    assert d._resolve_sensitive_features_for_batch(batch, split="train") is None


def test_method_signature_detection_and_optional_sensitive_calling():
    d = _DummyMixin()

    def with_sensitive(x, sensitive_features=None):
        return (x, sensitive_features)

    def with_kwargs(x, **kwargs):
        return (x, kwargs.get("sensitive_features"))

    def plain(x):
        return x

    assert d._method_accepts_sensitive_features(with_sensitive)
    assert d._method_accepts_sensitive_features(with_kwargs)
    assert not d._method_accepts_sensitive_features(plain)

    assert d._call_with_optional_sensitive(with_sensitive, 1, "s") == (1, "s")
    assert d._call_with_optional_sensitive(plain, 1, "s") == 1


def test_fit_defended_estimator_paths():
    d = _DummyMixin()
    data = _dummy_data()

    class FitWithSensitive:
        def __init__(self):
            self.calls = []

        def fit(self, x, y, sensitive_features=None):
            self.calls.append((x, y, sensitive_features))
            return self

    class FitPlain:
        def __init__(self):
            self.calls = []

        def fit(self, x, y):
            self.calls.append((x, y))
            return self

    f1 = FitWithSensitive()
    out1 = d._fit_defended_estimator(f1, data)
    assert out1 is f1
    assert len(f1.calls) == 1
    assert f1.calls[0][2] is not None

    f2 = FitPlain()
    out2 = d._fit_defended_estimator(f2, data)
    assert out2 is f2
    assert len(f2.calls) == 1

    sentinel = object()
    assert d._fit_defended_estimator(sentinel, None) is sentinel


def test_torch_device_resolve_and_move_paths(monkeypatch):
    d = _DummyMixin()

    monkeypatch.setattr("deckard.model.fairness.resolve_torch_device", lambda _: (_ for _ in ()).throw(RuntimeError("x")))
    assert d._resolve_torch_device("cpu") is None

    assert d._move_torch_model_to_device(model_obj=object(), requested_device=None) is not None

    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)

    monkeypatch.setattr(d, "_resolve_torch_device", lambda _: (_ for _ in ()).throw(RuntimeError("bad dev")))
    moved = d._move_torch_model_to_device(model_obj=model, requested_device="cpu")
    assert moved is model


def test_resolve_fairlearn_model_param_paths(monkeypatch):
    d = _DummyMixin()

    assert d._resolve_fairlearn_model_param(None, fallback="f") == "f"

    class GoodSpec:
        def get_model(self):
            return "model"

    assert d._resolve_fairlearn_model_param(GoodSpec()) == "model"

    class BadSpec:
        def get_model(self):
            raise RuntimeError("no")

        _model = "cached"

    assert d._resolve_fairlearn_model_param(BadSpec()) == "cached"

    monkeypatch.setattr("deckard.model.fairness.load_class", lambda target, **kwargs: {"target": target, **kwargs})

    resolved_dict = d._resolve_fairlearn_model_param({"model_type": "m.Target", "model_params": {"a": 1}})
    assert resolved_dict["target"] == "m.Target"

    resolved_named = d._resolve_fairlearn_model_param({"name": "x.Y", "p": 2})
    assert resolved_named["target"] == "x.Y"
    assert resolved_named["p"] == 2

    assert d._resolve_fairlearn_model_param("pkg.Object")["target"] == "pkg.Object"
    assert d._resolve_fairlearn_model_param("literal") == "literal"


def test_apply_fairlearn_defense_error_paths(monkeypatch):
    cfg = FairlearnModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)

    cfg.defense_name = None
    with pytest.raises(ValueError, match="requires a fairlearn defense_name"):
        cfg._apply_fairlearn_defense(_dummy_data())

    cfg.defense_name = "fairlearn.reductions.ExponentiatedGradient"
    cfg._model = None
    with pytest.raises(ValueError, match="must have a fitted estimator"):
        cfg._apply_fairlearn_defense(_dummy_data())

    cfg._model = object()
    monkeypatch.setattr("deckard.model.fairness.resolve_class", lambda _: (_ for _ in ()).throw(ImportError("missing")))
    with pytest.raises(ImportError, match="Could not import defense class"):
        cfg._apply_fairlearn_defense(_dummy_data())


def test_apply_fairlearn_defense_constraints_and_submodule_paths(monkeypatch):
    data = _dummy_data()
    cfg = FairlearnModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
    cfg._model = object()
    cfg.get_model = lambda: object()

    class DummyDefense:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(self, x, y, sensitive_features=None):
            _ = (x, y, sensitive_features)
            return self

    monkeypatch.setattr("deckard.model.fairness.resolve_class", lambda _: DummyDefense)
    monkeypatch.setattr("deckard.model.fairness.load_class", lambda target, **kwargs: {"target": target, **kwargs})

    cfg.defense_name = "fairlearn.reductions.ExponentiatedGradient"
    cfg.defense_params = {"constraints": {"_target_": "pkg.Constraint", "eps": 0.1}}
    out = cfg._apply_fairlearn_defense(data)
    assert out is not None

    cfg.defense_name = "fairlearn.postprocessing.ThresholdOptimizer"
    cfg.defense_params = {}
    out2 = cfg._apply_fairlearn_defense(data)
    assert out2 is not None

    cfg.defense_name = "fairlearn.unknown.Something"
    cfg.defense_params = {}
    with pytest.raises(NotImplementedError):
        cfg._apply_fairlearn_defense(data)

    cfg.defense_name = "fairlearn.reductions.ExponentiatedGradient"
    cfg.defense_params = {}
    with pytest.raises(ValueError, match="require a 'constraints'"):
        cfg._apply_fairlearn_defense(data)


def test_apply_fairlearn_defense_adversarial_path(monkeypatch):
    data = _dummy_data()
    cfg = FairlearnModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
    cfg._model = object()
    cfg.get_model = lambda: "base"

    class DummyDefense:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x, y, sensitive_features=None):
            _ = (x, y, sensitive_features)
            return self

    monkeypatch.setattr("deckard.model.fairness.resolve_class", lambda _: DummyDefense)
    monkeypatch.setattr(cfg, "_resolve_fairlearn_model_param", lambda spec, fallback=None: spec or fallback)
    monkeypatch.setattr(cfg, "_adapt_binary_torch_predictor", lambda predictor, data: f"adapted-{predictor}")

    cfg.defense_name = "fairlearn.adversarial.AdversarialFairnessClassifier"
    cfg.defense_params = {"predictor_model": "pred", "adversary_model": "adv"}
    out = cfg._apply_fairlearn_defense(data)
    assert out.kwargs["predictor_model"] == "adapted-pred"
    assert out.kwargs["adversary_model"] == "adv"


def test_train_predict_predict_proba_paths(monkeypatch):
    cfg = FairlearnModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
    cfg.data = _dummy_data()
    cfg._model = None

    with pytest.raises(ValueError, match="Model not initialized"):
        cfg._train(pd.DataFrame({"x": [1]}), pd.Series([0]))

    class Model:
        def __init__(self):
            self.fit_calls = []
            self.predict_calls = []

        def fit(self, x, y, sensitive_features=None, **kwargs):
            self.fit_calls.append((x, y, sensitive_features, kwargs))

        def predict(self, x, sensitive_features=None):
            self.predict_calls.append((x, sensitive_features))
            if isinstance(x, pd.DataFrame):
                raise TypeError("loop of ufunc does not support argument")
            return np.array([0, 1])

        def predict_proba(self, x, sensitive_features=None):
            _ = (x, sensitive_features)
            return np.array([[0.4, 0.6], [0.8, 0.2]])

    cfg._model = Model()
    cfg.fit_params = {}
    cfg._train(cfg.data.X_train, cfg.data.y_train)
    pred = cfg._predict(cfg.data.X_test)
    assert pred.shape[0] == 2

    cfg.probability = False
    with pytest.raises(ValueError, match="does not support probability"):
        cfg._predict_proba(cfg.data.X_test)

    cfg.probability = True
    proba = cfg._predict_proba(cfg.data.X_test)
    assert proba.shape == (2, 2)


def test_resolve_sensitive_features_and_normalization_branches(monkeypatch):
    cfg = FairlearnModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
    cfg.data = _dummy_data()

    y = pd.Series([0, 1])
    resolved = cfg._resolve_sensitive_features(y, mode="test")
    assert list(resolved) == ["a", "b"]

    monkeypatch.setattr(pd.Series, "reindex", lambda self, idx: (_ for _ in ()).throw(RuntimeError("x")))
    assert cfg._resolve_sensitive_features(y, mode="test") is None


def test_compute_sensitive_fairness_scores_prediction_shape_paths():
    cfg = FairlearnModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
    cfg.data = _dummy_data()

    y_true = pd.Series([0, 1])

    with pytest.raises(ValueError, match="Unsupported prediction shape"):
        cfg._compute_sensitive_fairness_scores(y_true, np.zeros((2, 2, 2)))

    scores_bin = cfg._compute_sensitive_fairness_scores(y_true, np.array([[0.2], [0.9]]))
    assert "sensitive_feature_accuracy_difference" in scores_bin

    scores_multi = cfg._compute_sensitive_fairness_scores(y_true, np.array([[0.9, 0.1], [0.2, 0.8]]))
    assert "a_accuracy" in scores_multi


def test_group_fairness_and_fallback_apply_defense(monkeypatch):
    cfg = FairlearnModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
    cfg.data = _dummy_data()

    y_true = pd.Series([0, 1])
    y_pred = pd.Series([0, 1])
    out = cfg._compute_group_fairness_scores(y_true, y_pred)
    assert isinstance(out, dict)

    defense = FairlearnDefenseConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
    monkeypatch.setattr("deckard.model.fairness.DefenseConfig.apply_defense", lambda self, data: "base")
    defense.defense_name = "art.defences.postprocessor.ClassLabels"
    assert defense.apply_defense(_dummy_data()) == "base"

    defense.defense_name = "fairlearn.postprocessing.ThresholdOptimizer"
    monkeypatch.setattr(defense, "_apply_fairlearn_defense", lambda data: "fair")
    assert defense.apply_defense(_dummy_data()) == "fair"


def test_fairlearn_pytorch_overrides_delegate(monkeypatch):
    cfg = FairlearnPytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 2, "out_features": 1},
        classifier=True,
    )

    monkeypatch.setattr("deckard.model.fairness.PytorchModelConfig._train", lambda self, x, y: "trained")
    monkeypatch.setattr("deckard.model.fairness.PytorchModelConfig._predict", lambda self, x: "pred")

    assert cfg._train(None, None) == "trained"
    assert cfg._predict(None) == "pred"


def test_is_torch_module_model_and_import_fallback(monkeypatch):
    d = _DummyMixin()
    d._model = None
    assert d._is_torch_module_model() is False

    d._model = object()
    assert d._is_torch_module_model() is False

    torch = pytest.importorskip("torch")
    d._model = torch.nn.Linear(2, 1)
    assert d._is_torch_module_model() is True

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert d._is_torch_module_model() is False


def test_infer_and_resolve_sensitive_none_and_no_match_paths():
    d = _DummyMixin()
    assert d._infer_split_from_batch(pd.DataFrame({"x": [1]})) is None

    d.data = _dummy_data()
    foreign = pd.DataFrame({"x": [99]}, index=[999])
    assert d._infer_split_from_batch(foreign) is None
    assert d._resolve_sensitive_features_for_batch(foreign) is None


def test_method_accepts_sensitive_handles_signature_errors():
    d = _DummyMixin()
    assert d._method_accepts_sensitive_features(3) is False


def test_move_torch_model_to_device_import_error_and_non_module(monkeypatch):
    d = _DummyMixin()
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert d._move_torch_model_to_device(object(), requested_device="cpu") is not None


def test_resolve_fairlearn_model_param_target_get_model_failure_and_configbase(monkeypatch):
    d = _DummyMixin()

    class ObjWithBadGetModel:
        def get_model(self):
            raise RuntimeError("fail")

    monkeypatch.setattr("deckard.model.fairness.load_class", lambda target, **kwargs: ObjWithBadGetModel())
    obj = d._resolve_fairlearn_model_param({"_target_": "pkg.Target"})
    assert isinstance(obj, ObjWithBadGetModel)

    class Cfg(ConfigBase):
        def __init__(self):
            self.model_type = "pkg.Model"
            self.model_params = {"p": 1}

        def to_dict(self, for_hash=False):
            _ = for_hash
            return {"model_type": self.model_type, "model_params": self.model_params}

    monkeypatch.setattr("deckard.model.fairness.load_class", lambda target, **kwargs: {"target": target, **kwargs})
    resolved = d._resolve_fairlearn_model_param(Cfg())
    assert resolved["target"] == "pkg.Model"
    assert resolved["p"] == 1


def test_adapt_binary_torch_predictor_branches_and_errors():
    d = _DummyMixin()
    torch = pytest.importorskip("torch")
    data = SimpleNamespace(
        y_train=torch.tensor([0, 1, 1, 0]),
        X_train=torch.randn(4, 2),
    )

    class TwoLogit(torch.nn.Module):
        num_classes = 2

        def forward(self, x):
            return torch.randn(x.shape[0], 2)

    adapted = d._adapt_binary_torch_predictor(TwoLogit(), data)
    out = adapted(torch.randn(3, 2))
    assert out.shape == (3, 1)

    class OneLogit(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 1)

    adapted_one = d._adapt_binary_torch_predictor(OneLogit(), data)
    assert adapted_one(torch.randn(2, 2)).shape == (2, 1)

    class BadShape(torch.nn.Module):
        num_classes = 2

        def forward(self, x):
            return torch.randn(x.shape[0], 2, 2)

    adapted_bad = d._adapt_binary_torch_predictor(BadShape(), data)
    with pytest.raises(ValueError, match="Unsupported predictor output shape"):
        adapted_bad(torch.randn(2, 2))


def test_resolve_fairness_defense_spec_and_constraints_missing_target(monkeypatch):
    cfg = FairlearnModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
    cfg._model = object()
    cfg.get_model = lambda: object()
    cfg.defense_name = "fairlearn.reductions.ExponentiatedGradient"
    cfg.defense_params = {"constraints": {"eps": 0.1}}

    class DummyDefense:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        def fit(self, x, y, sensitive_features=None):
            _ = (x, y, sensitive_features)
            return self

    monkeypatch.setattr("deckard.model.fairness.resolve_class", lambda _: DummyDefense)
    with pytest.raises(ValueError, match="constraints dict must include '_target_'"):
        cfg._apply_fairlearn_defense(_dummy_data())

    d = _DummyMixin()
    d.defense = SimpleNamespace(defense_name="fairlearn.postprocessing.ThresholdOptimizer", defense_params={"x": 1})
    assert d._resolve_fairness_defense_spec()[0].startswith("fairlearn.")


def test_predict_and_predict_proba_model_none_paths():
    cfg = FairlearnModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
    cfg._model = None
    with pytest.raises(ValueError, match="Model not initialized"):
        cfg._predict(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="Model not initialized"):
        cfg._predict_proba(pd.DataFrame({"x": [1]}))


def test_resolve_sensitive_features_reindex_exception_and_prediction_threshold_edges(monkeypatch):
    cfg = FairlearnModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
    cfg.data = _dummy_data()

    original_reindex = pd.Series.reindex
    y = pd.Series([0, 1], index=[100, 101])
    monkeypatch.setattr(pd.Series, "reindex", lambda self, idx: (_ for _ in ()).throw(RuntimeError("x")))
    assert cfg._resolve_sensitive_features(y, mode="test") is None
    monkeypatch.setattr(pd.Series, "reindex", original_reindex)

    cfg2 = FairlearnModelConfig(model_type="sklearn.linear_model.LogisticRegression", classifier=True)
    cfg2.data = _dummy_data()
    y_numeric = pd.Series([0, 1])
    scores_threshold0 = cfg2._compute_sensitive_fairness_scores(y_numeric, np.array([[-1.0], [2.0]]))
    assert "sensitive_feature_accuracy_ratio" in scores_threshold0

    y_text = pd.Series(["no", "yes"])
    with pytest.raises(ValueError, match="Mix of label input types"):
        cfg2._compute_sensitive_fairness_scores(y_text, np.array([[0.1], [0.9]]))
