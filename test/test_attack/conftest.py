import pytest
import numpy as np
import pandas as pd


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
def require_fairlearn():
    """Skip tests that require optional fairlearn dependency."""
    return pytest.importorskip("fairlearn")


@pytest.fixture
def require_torch():
    """Skip tests that require optional torch dependency."""
    return pytest.importorskip("torch")
