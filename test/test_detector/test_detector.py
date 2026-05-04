from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_classification

from deckard.detector import DetectorConfig


class _FakeBinaryInputDetector:
    def __init__(self, detector, **kwargs):
        self.detector = detector
        self.kwargs = kwargs

    def fit(self, x, y, **kwargs):
        _ = x, y, kwargs

    def detect(self, x, batch_size=128, **kwargs):
        _ = batch_size, kwargs
        n = len(x) // 2
        preds = np.concatenate([np.zeros(n, dtype=int), np.ones(len(x) - n, dtype=int)])
        return {"mock": True}, preds


def _make_data_and_attack(n=8, d=4):
    data = SimpleNamespace(
        X_train=np.random.rand(n, d).astype(np.float32),
        y_train=np.random.randint(0, 2, size=(n,)),
        X_test=np.random.rand(n, d).astype(np.float32),
        y_test=np.random.randint(0, 2, size=(n,)),
    )
    attack = SimpleNamespace(
        attack_predictions=np.random.rand(n, d).astype(np.float32),
    )
    return data, attack


def test_detector_requires_attack_predictions():
    data, _ = _make_data_and_attack()
    detector = DetectorConfig(
        detector_model={
            "model_type": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 20},
        },
    )
    try:
        detector(data=data, model=None, attack=SimpleNamespace(attack_predictions=None))
        assert False, "Expected ValueError for missing attack_predictions"
    except ValueError:
        assert True


def test_detector_runs_and_emits_scores_with_mock_detector():
    data, attack = _make_data_and_attack()
    detector = DetectorConfig(
        detector_model={
            "model_type": "sklearn.linear_model.LogisticRegression",
            "classifier": True,
            "model_params": {"max_iter": 20},
        },
        fit_params={"batch_size": 4},
    )

    with patch("deckard.detector.base.resolve_class", return_value=_FakeBinaryInputDetector):
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

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 2)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

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
    y_train, y_test = y[:80], y[80:]

    victim_model = TinyNet()
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

    detector_model = TinyNet()
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

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 2)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

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

    X_mix = np.vstack([X_train, X_poison]).astype(np.float32)
    y_mix = np.concatenate([y_train, y_poison.astype(np.int64)])

    model = TinyNet()
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

    is_clean = np.asarray(is_clean).reshape(-1)
    assert isinstance(report, dict)
    assert len(is_clean) == len(X_mix)
