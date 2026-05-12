import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Make test-local helpers importable regardless of --import-mode
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def set_test_max_samples_env(monkeypatch):
    """Keep dataset sizes bounded in tests without exposing a runtime config knob."""
    monkeypatch.setenv("DECKARD_TEST_MAX_SAMPLES", "200")


# Auto-mark expensive tests
def pytest_collection_modifyitems(config, items):
    """Automatically mark expensive tests with @pytest.mark.slow."""
    expensive_files = {
        "test_pytorch_serialization.py",
        "test_pytorch_experiment.py",
        "test_pytorch.py",
        "test_pytorch_fairness_integration.py",
        "test_pytorch_anjana_integration.py",
    }

    for item in items:
        # Mark PyTorch-related test files
        if any(exp_file in str(item.fspath) for exp_file in expensive_files):
            item.add_marker(pytest.mark.slow)

        # Mark tests that involve expensive operations
        if "optimize" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


class TinyData:
    """Minimal test dataset with isolated RNG per instance."""

    def __init__(self, seed=11):
        self.rng = np.random.default_rng(seed)
        self.X_train = pd.DataFrame(
            {
                "feature": self.rng.normal(size=40),
                "sensitive": self.rng.integers(0, 2, size=40),
                "other": self.rng.normal(size=40),
            },
        )
        self.y_train = pd.Series(
            (self.X_train["feature"] + self.X_train["other"] > 0).astype(int),
            name="target",
        )
        self.X_test = pd.DataFrame(
            {
                "feature": self.rng.normal(size=24),
                "sensitive": self.rng.integers(0, 2, size=24),
                "other": self.rng.normal(size=24),
            },
        )
        self.y_test = pd.Series(
            (self.X_test["feature"] + self.X_test["other"] > 0).astype(int),
            name="target",
        )


@pytest.fixture
def tiny_data():
    """Fixture providing isolated TinyData instance."""
    return TinyData()


# Dummy classes for test_optimize.py and related tests
class DummyStudy:
    """Mock Optuna Study."""

    def __init__(self):
        self.metric_names = None
        self.user_attrs = {}

    def set_metric_names(self, names):
        self.metric_names = list(names)

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class DummyFiles:
    """Mock Files config."""

    def __init__(self, tmp_path=None):
        self.experiment_name = None
        self.post_init_calls = 0
        if tmp_path is not None:
            self.log_file = str(tmp_path / "run.log")
            self.score_file = str(tmp_path / "scores.json")
            self.params_file = str(tmp_path / "params.yaml")
            self.error_file = str(tmp_path / "error.log")

    def __post_init__(self):
        self.post_init_calls += 1

    def to_dict(self):
        return {
            "log_file": self.log_file,
            "score_file": self.score_file,
            "params_file": self.params_file,
            "error_file": self.error_file,
        }

    def _get_file_dict(self):
        return self.to_dict()


class DummyConf:
    """Mock Conf config."""

    def __init__(self):
        self.files = DummyFiles()
        self.experiment_name = None
        self.post_init_calls = 0

    def __post_init__(self):
        self.post_init_calls += 1


class DummyStorage:
    """Mock Optuna Storage."""

    def __init__(self):
        self.attrs = {}

    def set_trial_user_attr(self, trial_id, key, value):
        self.attrs[(trial_id, key)] = value


class DummyTrial:
    """Mock Optuna Trial."""

    def __init__(self, number, trial_id, user_attrs=None):
        self.number = number
        self._trial_id = trial_id
        self.user_attrs = user_attrs or {}


@pytest.fixture
def dummy_study():
    """Fixture providing DummyStudy instance."""
    return DummyStudy()


@pytest.fixture
def dummy_files(tmp_path):
    """Fixture providing DummyFiles instance with tmp_path."""
    return DummyFiles(tmp_path=tmp_path)


@pytest.fixture
def dummy_conf():
    """Fixture providing DummyConf instance."""
    return DummyConf()


@pytest.fixture
def dummy_storage():
    """Fixture providing DummyStorage instance."""
    return DummyStorage()


@pytest.fixture
def dummy_trial():
    """Fixture providing DummyTrial instance."""
    return DummyTrial(number=0, trial_id="test_trial")
