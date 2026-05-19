import pytest
from helpers import TinyFairness

TinyData = TinyFairness


@pytest.fixture
def require_fairlearn():
    """Skip tests that require optional fairlearn dependency."""
    return pytest.importorskip("fairlearn")
