"""Shared test helpers for subprocess-based integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


class TinyFairness:
    """Minimal fairness dataset example with explicit sensitive features."""

    def __init__(self, seed: int = 11) -> None:
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
        self.sensitive_train = self.X_train["sensitive"].reset_index(drop=True)
        self.sensitive_test = self.X_test["sensitive"].reset_index(drop=True)
        self._sensitive_train = self.sensitive_train
        self._sensitive_test = self.sensitive_test


TinyData = TinyFairness


def load_env_from_deckard_rc(path: Path) -> dict[str, str]:
    """Parse ``export KEY=VALUE`` lines from a ``.deckard_rc`` file."""
    env_overrides: dict[str, str] = {}
    if not path.exists():
        return env_overrides
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        key_value = line[len("export ") :]
        if "=" not in key_value:
            continue
        key, value = key_value.split("=", 1)
        env_overrides[key.strip()] = value.strip().strip('"').strip("'")
    return env_overrides


def make_runtime_env(rc_path: Path) -> dict[str, str]:
    """Return an os.environ copy augmented with rc file vars and test defaults."""
    env = os.environ.copy()
    env.update(load_env_from_deckard_rc(rc_path))
    env.setdefault("DECKARD_TEST_MAX_SAMPLES", "200")
    env.setdefault("MPLBACKEND", "Agg")
    return env


def reset_hydra_state() -> None:
    """Clear Hydra's global state between compose calls."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def load_canonical_data_profile(profile_name: str, framework: str = "sklearn") -> dict:
    """Load a canonical data profile from examples/<framework>/config/data."""
    repo_root = Path(__file__).resolve().parents[1]
    profile_path = (
        repo_root / "examples" / framework / "config" / "data" / f"{profile_name}.yaml"
    )
    cfg = OmegaConf.load(profile_path)
    return OmegaConf.to_container(cfg, resolve=True)
