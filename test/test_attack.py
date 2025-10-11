import pandas as pd
import numpy as np
from deckard.attack import AttackConfig, attack_defaults


def test_attackconfig_default_initialization():
    config = AttackConfig()
    assert isinstance(config, AttackConfig)
    assert config.attack_name == "art.attacks.evasion.HopSkipJump"
    assert config.attack_size == 10
    assert config._attack is None
    assert isinstance(config._score_dict, dict)


def test_attackconfig_hash_and_post_init():
    config = AttackConfig(
        attack_name="art.attacks.evasion.FastGradientMethod",
        attack_params={"eps": 0.2},
    )
    h = hash(config)
    assert isinstance(h, int)
    assert config._attack is None
    assert isinstance(config._score_dict, dict)


def test_attack_defaults_keys():
    expected_keys = [
        "blackbox_evasion",
        "whitebox_evasion",
        "blackbox_attribute_inference",
        "whitebox_attribute_inference",
        "blackbox_membership_inference",
    ]
    for key in expected_keys:
        assert key in attack_defaults


def test_attackconfig_pop_attribute_removes_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    config = AttackConfig()
    arr, col = config._pop_attribute(df, "a")
    assert isinstance(arr, (list, pd.DataFrame, pd.Series, type(df.values)))
    assert col.name == "a"
    assert all(col == pd.Series([1, 2, 3]))


def test_attackconfig_score_attack_sets_score_dict():
    config = AttackConfig()
    ben_pred = np.array([0, 1, 1, 0])
    adv_pred = np.array([1, 1, 0, 0])
    y_true = np.array([0, 1, 1, 0])
    config._score_attack(ben_pred, adv_pred, y_true)
    assert isinstance(config._score_dict, dict)
    for key in [
        "adversarial_accuracy",
        "adversarial_precision",
        "adversarial_recall",
        "adversarial_f1-score",
        "adversarial_success_rate",
    ]:
        assert key in config._score_dict


def test_attackconfig_get_benign_preds_shape():
    class DummyEstimator:
        def predict(self, X):
            return np.zeros((len(X), 2))

    class DummyData:
        def __call__(self, *args, **kwargs):
            X_train = pd.DataFrame(np.random.rand(20, 3), columns=["a", "b", "c"])
            y_train = pd.Series([0] * 10 + [1] * 10)
            X_test = pd.DataFrame(np.random.rand(20, 3), columns=["a", "b", "c"])
            y_test = pd.Series([0] * 10 + [1] * 10)
            return X_train, y_train, X_test, y_test

    config = AttackConfig(attack_size=5)
    n, ben_pred_labels, X_subset, y_subset = config._get_benign_preds(
        DummyData(),
        DummyEstimator(),
        train=False,
    )
    assert n == 5
    assert len(ben_pred_labels) == 5
    assert X_subset.shape[0] == 5
    assert len(y_subset) == 5


def test_attackconfig_get_feature_vector_preds_shape():
    class DummyData:
        def __call__(self, targeted_attribute):
            X_train = pd.DataFrame(np.random.rand(20, 3), columns=["a", "b", "c"])
            y_train = pd.Series([0] * 10 + [1] * 10)
            a_train = pd.Series(["x"] * 10 + ["y"] * 10)
            X_test = pd.DataFrame(np.random.rand(20, 3), columns=["a", "b", "c"])
            y_test = pd.Series([0] * 10 + [1] * 10)
            a_test = pd.Series(["x"] * 10 + ["y"] * 10)
            return X_train, y_train, a_train, X_test, y_test, a_test

    config = AttackConfig(attack_size=5)
    n, X_subset, y_subset, a_subset = config._get_feature_vector_preds(
        DummyData(),
        "a",
        train=False,
    )
    assert n == 5
    assert X_subset.shape[0] == 5
    assert len(y_subset) == 5
    assert len(a_subset) == 5
