from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf
from sklearn.datasets import make_classification

from deckard.detector import DetectorConfig
from deckard.detector.canon import (
    ensure_detector_runtime_contract,
    normalize_detector_runtime_split_mode,
    normalize_detector_score_mode,
    normalize_detector_score_stage,
    normalize_detector_stage,
)
from deckard.detector.default import DetectorScorerConfig


class _FakeBinaryInputDetector:
    def __init__(self, detector, **kwargs):
        self.detector = detector
        self.kwargs = kwargs

    def fit(self, x, y, **kwargs):
        _ = x, y, kwargs

    def detect(self, x, batch_size=128, **kwargs):
        _ = batch_size, kwargs
        n = len(x) // 2
        preds = np.concatenate(
            [np.zeros(n, dtype=int), np.ones(len(x) - n, dtype=int)],
        )
        return {"mock": True}, preds


class _TinyNet:
    def __init__(self, nn, in_features=8, hidden=16, out_features=2):
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )


def _make_data_and_attack(n=8, d=4, seed=0):
    rng = np.random.default_rng(seed)
    data = SimpleNamespace(
        X_train=rng.random((n, d), dtype=np.float32),
        y_train=rng.integers(0, 2, size=(n,)),
        X_test=rng.random((n, d), dtype=np.float32),
        y_test=rng.integers(0, 2, size=(n,)),
    )
    attack = SimpleNamespace(
        attack_predictions=rng.random((n, d), dtype=np.float32),
    )
    return data, attack


def test_detector_requires_attack_predictions():
    data, _ = _make_data_and_attack()
    detector = DetectorConfig(
        detector_model={
            "name": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 20},
        },
    )
    with pytest.raises(ValueError):
        detector(
            data=data,
            model=None,
            attack=SimpleNamespace(attack_predictions=None),
        )


def test_detector_runs_and_emits_scores_with_mock_detector():
    data, attack = _make_data_and_attack()
    detector = DetectorConfig(
        detector_model={
            "name": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 20},
        },
        fit_params={"batch_size": 4},
    )

    with patch(
        "deckard.detector.base.resolve_class",
        return_value=_FakeBinaryInputDetector,
    ):
        scores = detector(data=data, model=None, attack=attack)

    assert "detector_accuracy" in scores
    assert "detector_precision" in scores
    assert "detector_recall" in scores
    assert "detector_f1" in scores
    assert scores["detector_n"] == 2 * len(attack.attack_predictions)


def test_real_art_evasion_binary_input_detector_executes():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    optim = pytest.importorskip("torch.optim")
    PyTorchClassifier = pytest.importorskip(
        "art.estimators.classification",
    ).PyTorchClassifier
    FastGradientMethod = pytest.importorskip("art.attacks.evasion").FastGradientMethod
    BinaryInputDetector = pytest.importorskip(
        "art.defences.detector.evasion",
    ).BinaryInputDetector
    torch.manual_seed(42)

    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X_train, X_test = X[:80], X[80:]
    y_train = y[:80]

    victim_model = _TinyNet(nn).model
    victim = PyTorchClassifier(
        model=victim_model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(victim_model.parameters(), lr=0.01),
        input_shape=(8,),
        nb_classes=2,
        clip_values=(-5.0, 5.0),
    )
    victim.fit(X_train, y_train, nb_epochs=2, batch_size=16)

    attack = FastGradientMethod(estimator=victim, eps=0.2)
    X_adv = attack.generate(X_test)
    assert np.any(np.abs(X_adv - X_test) > 1e-8)

    detector_model = _TinyNet(nn).model
    detector_backend = PyTorchClassifier(
        model=detector_model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(detector_model.parameters(), lr=0.01),
        input_shape=(8,),
        nb_classes=2,
        clip_values=(-5.0, 5.0),
    )
    detector = BinaryInputDetector(detector=detector_backend)

    X_det = np.vstack([X_test, X_adv]).astype(np.float32)
    y_det = np.concatenate(
        [
            np.zeros(len(X_test), dtype=np.int64),
            np.ones(len(X_adv), dtype=np.int64),
        ],
    )
    detector.fit(X_det, y_det, nb_epochs=2, batch_size=16)
    report, is_adv = detector.detect(X_det, batch_size=16)

    is_adv = np.asarray(is_adv).reshape(-1).astype(int)
    assert isinstance(report, dict)
    assert len(is_adv) == len(y_det)
    assert float(np.mean(is_adv == y_det)) >= 0.0


def test_real_art_poisoning_detector_executes():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    optim = pytest.importorskip("torch.optim")
    PyTorchClassifier = pytest.importorskip(
        "art.estimators.classification",
    ).PyTorchClassifier
    PoisoningAttackBackdoor = pytest.importorskip(
        "art.attacks.poisoning",
    ).PoisoningAttackBackdoor
    SpectralSignatureDefense = pytest.importorskip(
        "art.defences.detector.poison",
    ).SpectralSignatureDefense
    torch.manual_seed(7)

    X, y = make_classification(
        n_samples=100,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=2,
        random_state=7,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X_train, y_train = X[:70], y[:70]

    poison_idx = np.arange(14)
    poison_attack = PoisoningAttackBackdoor(
        lambda x: np.clip(x + 0.3, -5.0, 5.0),
    )
    X_poison, y_poison = poison_attack.poison(
        X_train[poison_idx],
        y=(1 - y_train[poison_idx]),
    )
    assert len(X_poison) == len(poison_idx)

    X_mix = np.vstack([X_train, X_poison]).astype(np.float32)
    y_mix = np.concatenate([y_train, y_poison.astype(np.int64)])

    model = _TinyNet(nn).model
    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.01),
        input_shape=(8,),
        nb_classes=2,
        clip_values=(-5.0, 5.0),
    )
    classifier.fit(X_mix, y_mix, nb_epochs=2, batch_size=16)

    detector = SpectralSignatureDefense(
        classifier=classifier,
        x_train=X_mix,
        y_train=y_mix,
        expected_pp_poison=0.2,
        batch_size=16,
    )
    report, is_clean = detector.detect_poison()

    is_clean = np.asarray(is_clean).reshape(-1).astype(int)
    assert isinstance(report, dict)
    assert len(is_clean) == len(X_mix)
    assert np.all((is_clean == 0) | (is_clean == 1))


def test_detector_model_coercion_dictconfig_and_invalid_type():
    cfg = DetectorConfig(
        detector_model=OmegaConf.create(
            {
                "name": "sklearn.linear_model.LogisticRegression",
                "classifier": True,
                "model_params": {"max_iter": 10},
            },
        ),
    )
    assert cfg.detector_model is not None

    with pytest.raises(TypeError, match="Unsupported detector_model type"):
        DetectorConfig(detector_model=123)


def test_detector_model_coercion_from_yaml_string(monkeypatch):
    class _DummyLoaded:
        def to_dict(self):
            return {
                "name": "sklearn.linear_model.LogisticRegression",
                "classifier": True,
                "model_params": {"max_iter": 10},
            }

    monkeypatch.setattr(
        "deckard.detector.base.ModelConfig.from_yaml",
        lambda _path: _DummyLoaded(),
    )

    cfg = DetectorConfig(detector_model="fake.yaml")
    assert cfg.detector_model is not None


def test_detector_canon_stage_and_score_aliases() -> None:
    assert normalize_detector_stage("before_fit") == "pre-fit"
    assert normalize_detector_stage(None) == "post-detect"
    assert normalize_detector_score_stage("pre_sample") == "pre-sample"
    assert normalize_detector_score_stage("auto") == "auto"
    assert normalize_detector_score_mode("evaluation") == "test"


def test_detector_canon_invalid_score_tokens_raise() -> None:
    with pytest.raises(ValueError, match="Unknown detector score stage"):
        normalize_detector_score_stage("definitely-not-a-stage")

    with pytest.raises(ValueError, match="Unknown detector score mode"):
        normalize_detector_score_mode("definitely-not-a-mode")


def test_detector_runtime_split_mode_supports_override_aliases() -> None:
    assert normalize_detector_runtime_split_mode("detect") == "test"
    assert (
        normalize_detector_runtime_split_mode(
            "holdout",
            aliases={"holdout": "val"},
        )
        == "val"
    )


def test_ensure_detector_runtime_contract_initializes_missing_attrs() -> None:
    runtime = SimpleNamespace()

    ensure_detector_runtime_contract(runtime)

    assert runtime.score_dict == {}
    assert runtime.detector is None
    assert runtime.detector_training_time is None
    assert runtime.detector_detection_time is None
    assert runtime.detector_predictions is None


@pytest.mark.parametrize(
    "alias",
    [DetectorConfig, DetectorScorerConfig],
    ids=["DetectorConfig", "DetectorScorerConfig"],
)
def test_detector_family_aliases_are_importable(alias):
    assert alias is not None


def test_detector_build_dataset_split_and_size_validation():
    data, attack = _make_data_and_attack(n=4, d=3)
    cfg = DetectorConfig(
        detector_model={
            "name": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 10},
        },
        fit_params={"split": "invalid"},
    )

    with pytest.raises(ValueError, match="Unsupported detector split"):
        cfg._build_detector_dataset(data, attack)

    empty_attack = SimpleNamespace(
        attack_predictions=np.empty((0, 3), dtype=np.float32),
    )
    cfg.fit_params = {"split": "test"}
    with pytest.raises(ValueError, match="must contain at least one"):
        cfg._build_detector_dataset(data, empty_attack)


def test_detector_build_backend_requires_detector_model():
    cfg = DetectorConfig(detector_model=None)
    with pytest.raises(ValueError, match="requires detector_model"):
        cfg._build_detector_backend(
            x_train=np.zeros((2, 2), dtype=np.float32),
            y_train=np.array([0, 1]),
        )


def test_detector_constructor_fallback_and_detect_poison_indices(monkeypatch):
    class _FallbackPoisonDetector:
        def __init__(self, classifier, x_train, y_train, **kwargs):
            _ = classifier, x_train, y_train, kwargs

        def fit(self, x, y, **kwargs):
            _ = x, y, kwargs

        def detect_poison(self, **kwargs):
            _ = kwargs
            # Return poison indices, not full-length clean mask.
            return {"mock": True}, np.array([0, 2])

    data, attack = _make_data_and_attack(n=4, d=2)
    cfg = DetectorConfig(
        detector_model={
            "name": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 10},
        },
        fit_params={"split": "test"},
    )

    monkeypatch.setattr(
        "deckard.detector.base.resolve_class",
        lambda _name: _FallbackPoisonDetector,
    )
    monkeypatch.setattr(
        DetectorConfig,
        "_build_detector_backend",
        lambda self, x_train, y_train: object(),
    )

    scores = cfg(data=data, model=None, attack=attack)
    assert scores["detector_n"] == 8
    assert "detector_accuracy" in scores


def test_detector_stage_normalization_tokens():
    assert normalize_detector_stage("before_fit") == "pre-fit"
    assert normalize_detector_stage("after_fit") == "post-fit"
    assert normalize_detector_stage("before_detect") == "pre-detect"
    assert normalize_detector_stage("after_detect") == "post-detect"


def test_detector_persists_files_and_score_state(tmp_path):
    data, attack = _make_data_and_attack(n=6, d=3)
    cfg = DetectorConfig(
        detector_model={
            "name": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 20},
        },
    )

    detector_file = tmp_path / "detector.pkl"
    detected_predictions_file = tmp_path / "detector_preds.pkl"
    score_file = tmp_path / "detector_scores.json"

    with patch(
        "deckard.detector.base.resolve_class",
        return_value=_FakeBinaryInputDetector,
    ):
        scores = cfg(
            data=data,
            model=None,
            attack=attack,
            detector_file=str(detector_file),
            detected_predictions_file=str(detected_predictions_file),
            score_file=str(score_file),
        )

    assert "detector_stage" in scores
    assert scores["detector_stage"] == "post-detect"
    assert scores["detector_execution_order"] == "post-attack"
    assert detector_file.exists()
    assert detected_predictions_file.exists()
    assert score_file.exists()


def test_detector_detect_poison_invalid_shape_raises(monkeypatch):
    class _BadPoisonDetector:
        def __init__(self, classifier, x_train, y_train, **kwargs):
            _ = classifier, x_train, y_train, kwargs

        def detect_poison(self, **kwargs):
            _ = kwargs
            return {"mock": True}, np.ones((2, 2), dtype=int)

    data, attack = _make_data_and_attack(n=3, d=2)
    cfg = DetectorConfig(
        detector_model={
            "name": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 10},
        },
    )

    monkeypatch.setattr(
        "deckard.detector.base.resolve_class",
        lambda _name: _BadPoisonDetector,
    )
    monkeypatch.setattr(
        DetectorConfig,
        "_build_detector_backend",
        lambda self, x_train, y_train: object(),
    )

    with pytest.raises(ValueError, match="Unsupported detect_poison output shape"):
        cfg(data=data, model=None, attack=attack)


def test_detector_raises_when_backend_has_no_detection_api(monkeypatch):
    class _NoDetectDetector:
        def __init__(self, detector, **kwargs):
            _ = detector, kwargs

    data, attack = _make_data_and_attack(n=3, d=2)
    cfg = DetectorConfig(
        detector_model={
            "name": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 10},
        },
    )

    monkeypatch.setattr(
        "deckard.detector.base.resolve_class",
        lambda _name: _NoDetectDetector,
    )
    monkeypatch.setattr(
        DetectorConfig,
        "_build_detector_backend",
        lambda self, x_train, y_train: object(),
    )

    with pytest.raises(
        AttributeError,
        match=r"exposes neither detect\(\) nor detect_poison\(\)",
    ):
        cfg(data=data, model=None, attack=attack)


class _ConfigLikeData:
    def __init__(self, n=6, d=3):
        self._rng = np.random.default_rng(41)
        self.n = n
        self.d = d

    def __call__(self):
        self.X_train = self._rng.random((self.n, self.d), dtype=np.float32)
        self.y_train = self._rng.integers(0, 2, size=(self.n,))
        self.X_test = self._rng.random((self.n, self.d), dtype=np.float32)
        self.y_test = self._rng.integers(0, 2, size=(self.n,))
        return {"data_ready": True}


class _ConfigLikeModel:
    def __init__(self):
        self._model = None
        self.train_calls = 0

    def __call__(self, data):
        self._model = object()
        _ = data
        return {"model_ready": True}

    def train(self, X, y):
        _ = X, y
        self.train_calls += 1


class _ConfigLikeAttack:
    name = "art.attacks.poisoning.BackdoorAttack"

    def __init__(self):
        self.attack_predictions = None

    def __call__(self, data, model):
        _ = model
        self.attack_predictions = np.array(data.X_test, copy=True)
        self.attack_predictions[0] = self.attack_predictions[0] + 5.0
        return {"attack_ready": True}


class _FixedDetectDetector:
    def __init__(self, detector, **kwargs):
        self.detector = detector
        self.kwargs = kwargs

    def fit(self, x, y, **kwargs):
        _ = x, y, kwargs

    def detect(self, x, batch_size=128, **kwargs):
        _ = batch_size, kwargs
        n = len(x) // 2
        pred = np.zeros(len(x), dtype=int)
        pred[n] = 1
        return {"mock": True}, pred


def test_detector_filter_mode_poison_updates_attack_and_retrains(monkeypatch):
    data = _ConfigLikeData(n=5, d=2)
    model = _ConfigLikeModel()
    attack = _ConfigLikeAttack()

    cfg = DetectorConfig(
        detector_model={
            "name": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 10},
        },
        mode="filter",
        filter_mode="poison",
        fit_params={"split": "test"},
    )

    monkeypatch.setattr(
        DetectorConfig,
        "_build_detector_backend",
        lambda self, x_train, y_train: object(),
    )
    monkeypatch.setattr(
        "deckard.detector.base.resolve_class",
        lambda _name: _FixedDetectDetector,
    )

    scores = cfg(data=data, model=model, attack=attack)

    assert model.train_calls == 1
    assert scores["poison_filter_success"] > 0.0
    assert "evasion_filter_success" in scores
    assert np.allclose(attack.attack_predictions[0], data.X_test[0])


def test_detector_filter_mode_evasion_sets_filtered_labels(monkeypatch):
    data, attack = _make_data_and_attack(n=6, d=3)
    attack.name = "art.attacks.evasion.FastGradientMethod"
    attack.attacked_labels = np.zeros(6, dtype=int)
    model = SimpleNamespace(_model=object())

    cfg = DetectorConfig(
        detector_model={
            "name": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 10},
        },
        mode="filter",
        filter_mode="evasion",
        fit_params={"split": "test"},
    )

    monkeypatch.setattr(
        DetectorConfig,
        "_build_detector_backend",
        lambda self, x_train, y_train: object(),
    )
    monkeypatch.setattr(
        "deckard.detector.base.resolve_class",
        lambda _name: _FixedDetectDetector,
    )

    scores = cfg(data=data, model=model, attack=attack)

    assert scores["evasion_filter_success"] > 0.0
    assert "poison_filter_success" in scores
    assert hasattr(attack, "filtered_attack_inputs")
    assert np.array_equal(attack.attacked_labels[:1], data.y_test[:1])


def test_detector_auto_filter_mode_uses_attack_name_field(monkeypatch):
    data = _ConfigLikeData(n=5, d=2)
    model = _ConfigLikeModel()
    attack = _ConfigLikeAttack()
    attack.name = "art.attacks.poisoning.BackdoorAttack"

    cfg = DetectorConfig(
        detector_model={
            "name": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 10},
        },
        mode="filter",
        filter_mode="auto",
        fit_params={"split": "test"},
    )

    monkeypatch.setattr(
        DetectorConfig,
        "_build_detector_backend",
        lambda self, x_train, y_train: object(),
    )
    monkeypatch.setattr(
        "deckard.detector.base.resolve_class",
        lambda _name: _FixedDetectDetector,
    )

    scores = cfg(data=data, model=model, attack=attack)
    assert scores["poison_filter_success"] > 0.0
