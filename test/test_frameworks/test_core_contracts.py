"""
Focused tests for deckard/frameworks/core.py abstract contracts.
Ensures that all core contract classes can be instantiated and enforce required interface methods.
"""
import pytest
import importlib
import inspect

CORE_CONTRACTS = [
    "deckard.frameworks.core.DataContractMixin",
    "deckard.frameworks.core.ModelContractMixin",
    "deckard.frameworks.core.AttackContractMixin",
    "deckard.frameworks.core.DetectorContractMixin",
    "deckard.frameworks.core.ExperimentContractMixin",
    "deckard.frameworks.core.ScorerContractMixin",
]

@pytest.mark.parametrize("qualified_name", CORE_CONTRACTS)
def test_core_contract_instantiation_and_methods(qualified_name):
    module_name, class_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    contract_cls = getattr(module, class_name)
    # Should be instantiable (may require no-arg constructor)
    instance = contract_cls()
    # Should have at least one abstract method or property
    abstract_methods = [
        name for name, value in inspect.getmembers(contract_cls)
        if getattr(value, "__isabstractmethod__", False)
    ]
    assert abstract_methods, f"{qualified_name} should define at least one abstract method or property"
