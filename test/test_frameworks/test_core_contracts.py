"""Focused tests for ``deckard.frameworks.core`` abstract contracts."""

import inspect

import pytest

from deckard.attack import AttackConfig
from deckard.data import DataConfig
from deckard.detector import DetectorConfig
from deckard.experiment import ExperimentConfig
from deckard.frameworks.core import (
    FrameworkAttackConfig,
    FrameworkDataConfig,
    FrameworkDetectorConfig,
    FrameworkExperimentConfig,
    FrameworkModelConfig,
    FrameworkScorerConfig,
)
from deckard.model import ModelConfig
from deckard.score import ScorerDictConfig


@pytest.mark.parametrize(
    "contract_cls",
    [
        FrameworkDataConfig,
        FrameworkModelConfig,
        FrameworkAttackConfig,
        FrameworkDetectorConfig,
        FrameworkExperimentConfig,
        FrameworkScorerConfig,
    ],
)
def test_framework_contracts_are_abstract(contract_cls):
    abstract_methods = sorted(getattr(contract_cls, "__abstractmethods__", set()))
    assert abstract_methods, f"{contract_cls.__name__} should define abstract API methods"
    assert inspect.isabstract(contract_cls), (
        f"{contract_cls.__name__} must remain abstract to enforce shared interface compliance"
    )


@pytest.mark.parametrize(
    ("runtime_cls", "contract_cls"),
    [
        (DataConfig, FrameworkDataConfig),
        (ModelConfig, FrameworkModelConfig),
        (AttackConfig, FrameworkAttackConfig),
        (DetectorConfig, FrameworkDetectorConfig),
        (ExperimentConfig, FrameworkExperimentConfig),
        (ScorerDictConfig, FrameworkScorerConfig),
    ],
)
def test_core_runtime_configs_implement_framework_contracts(runtime_cls, contract_cls):
    assert issubclass(runtime_cls, contract_cls), (
        f"{runtime_cls.__name__} should inherit {contract_cls.__name__} "
        "to enforce a shared framework interface"
    )
