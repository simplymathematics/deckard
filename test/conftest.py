import sys
from pathlib import Path

import pytest

# Make test-local helpers importable regardless of --import-mode
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def set_test_max_samples_env(monkeypatch):
    """Keep dataset sizes bounded in tests without exposing a runtime config knob."""
    monkeypatch.setenv("DECKARD_TEST_MAX_SAMPLES", "200")
