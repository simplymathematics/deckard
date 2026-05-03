import pytest


@pytest.fixture(autouse=True)
def set_test_max_samples_env(monkeypatch):
    """Keep dataset sizes bounded in tests without exposing a runtime config knob."""
    monkeypatch.setenv("DECKARD_TEST_MAX_SAMPLES", "200")
