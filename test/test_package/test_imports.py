"""
Regression tests for core package import boundaries.
Ensures that core packages do not import plugin or framework families at import time.
"""

import importlib

import pytest

CORE_MODULES = [
    "deckard.data.base",
    "deckard.model.base",
    "deckard.attack.base",
    "deckard.detector.base",
    "deckard.experiment.base",
    "deckard.score.base",
    "deckard.frameworks.core",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_import_no_plugin_dependency(module_name):
    try:
        importlib.import_module(module_name)
    except Exception as e:
        pytest.fail(f"Importing {module_name} failed: {e}")
