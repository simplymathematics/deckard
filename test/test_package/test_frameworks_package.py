import ast
import inspect
import pathlib

from deckard.frameworks import (
    DeclarativeConfigContract,
    FrameworkAttackConfig,
    FrameworkDataConfig,
    FrameworkDataPipelineConfig,
    FrameworkDataSamplerContract,
    FrameworkDetectorConfig,
    FrameworkExperimentConfig,
    FrameworkModelDefenseConfig,
    FrameworkModelConfig,
    FrameworkScorerConfig,
)
from dataclasses import dataclass

import pytest


def test_framework_contract_exports_are_importable():
    assert FrameworkDataConfig is not None
    assert FrameworkDataPipelineConfig is not None
    assert FrameworkDataSamplerContract is not None
    assert FrameworkModelConfig is not None
    assert FrameworkModelDefenseConfig is not None
    assert FrameworkAttackConfig is not None
    assert FrameworkDetectorConfig is not None
    assert FrameworkExperimentConfig is not None
    assert FrameworkScorerConfig is not None


@dataclass(eq=False, kw_only=True)
class _ContractStubBase:
    call_log: list[str] = None

    def _initialize_loading_behavior(self):
        self.call_log = ["init_loading"]

    def _initialize_persistence_behavior(self):
        self.call_log.append("init_persistence")

    def _initialize_scoring_behavior(self):
        self.call_log.append("init_scoring")

    def _initialize_context_behavior(self):
        self.call_log.append("init_context")

    def __call__(self, *args, **kwargs):
        return self.run_declared_execution(*args, **kwargs)

    def load_defaults(self):
        self.call_log.append("load_defaults")
        return "defaults"

    def load(self):
        self.call_log.append("load")
        return "load"

    def load_cached(self):
        self.call_log.append("load_cached")
        return "pretrained"

    def resolve_context(self, **context):
        self.call_log.append(f"resolve_context:{sorted(context.keys())}")
        return context

    def score(self, *args, **kwargs):
        _ = (args, kwargs)
        self.call_log.append("score")
        return {"score": 1}

    def save(self):
        self.call_log.append("save")
        return "saved"


@dataclass(eq=False, kw_only=True)
class _StubFrameworkDataConfig(_ContractStubBase, FrameworkDataConfig):
    def _validate_data_contract(self):
        self.call_log.append("validate")

    def load_data(self):
        self.call_log.append("load_data")
        return ("X", "y")

    def sample_data(self, X=None, y=None):
        _ = (X, y)
        self.call_log.append("sample_data")
        return ("X_train", "X_test", "y_train", "y_test")

    def fit_presample(self, X=None, y=None):
        _ = (X, y)
        self.call_log.append("fit_presample")
        return (X, y)

    def fit_X(self, X=None, y=None):
        _ = (X, y)
        self.call_log.append("fit_X")
        return (X, y)

    def fit_y(self, X=None, y=None):
        _ = (X, y)
        self.call_log.append("fit_y")
        return (X, y)

    def fit_Xy(self, X=None, y=None):
        _ = (X, y)
        self.call_log.append("fit_Xy")
        return (X, y)


@dataclass(eq=False, kw_only=True)
class _StubFrameworkDataPipelineConfig(_ContractStubBase, FrameworkDataPipelineConfig):
    def _validate_data_contract(self):
        self.call_log.append("validate")

    def load_data(self):
        self.call_log.append("load_data")
        return ("X", "y")

    def sample_data(self, X=None, y=None):
        _ = (X, y)
        self.call_log.append("sample_data")
        return ("X_train", "X_test", "y_train", "y_test")

    def fit_presample(self, X=None, y=None):
        _ = (X, y)
        self.call_log.append("fit_presample")
        return (X, y)

    def fit_X(self, X=None, y=None):
        _ = (X, y)
        self.call_log.append("fit_X")
        return (X, y)

    def fit_y(self, X=None, y=None):
        _ = (X, y)
        self.call_log.append("fit_y")
        return (X, y)

    def fit_Xy(self, X=None, y=None):
        _ = (X, y)
        self.call_log.append("fit_Xy")
        return (X, y)

    def build_pipeline(self):
        self.call_log.append("build_pipeline")
        return "pipeline"

    def run_pipeline(self, pipeline=None):
        self.call_log.append("run_pipeline")
        return pipeline


@dataclass(eq=False, kw_only=True)
class _StubFrameworkModelConfig(_ContractStubBase, FrameworkModelConfig):
    def _validate_model_contract(self):
        self.call_log.append("validate")

    def init_model(self, data=None):
        _ = data
        self.call_log.append("init_model")
        return data

    def fit_model(self, data=None):
        _ = data
        self.call_log.append("fit_model")
        return data


@dataclass(eq=False, kw_only=True)
class _StubFrameworkAttackConfig(_ContractStubBase, FrameworkAttackConfig):
    def _validate_attack_contract(self):
        self.call_log.append("validate")

    def build_attack(self, model=None, data=None):
        _ = (model, data)
        self.call_log.append("build_attack")
        return {"attack": "ok"}


@dataclass(eq=False, kw_only=True)
class _StubFrameworkDetectorConfig(_ContractStubBase, FrameworkDetectorConfig):
    def _validate_detector_contract(self):
        self.call_log.append("validate")

    def build_detector(self, model=None, attack=None):
        _ = (model, attack)
        self.call_log.append("build_detector")
        return {"detector": "ok"}


@dataclass(eq=False, kw_only=True)
class _StubFrameworkExperimentConfig(_ContractStubBase, FrameworkExperimentConfig):
    def _validate_experiment_contract(self):
        self.call_log.append("validate")

    def run_experiment(self):
        self.call_log.append("run_experiment")
        return {"experiment": "ok"}


@dataclass(eq=False, kw_only=True)
class _StubFrameworkScorerConfig(_ContractStubBase, FrameworkScorerConfig):
    def _validate_scorer_contract(self):
        self.call_log.append("validate")

    def score(self, *, data=None, model=None, attack=None):
        _ = (data, model, attack)
        self.call_log.append("score")
        return {"score": 1}


@dataclass(eq=False, kw_only=True)
class _DefenseContractStubBase:
    call_log: list[str] = None

    def _initialize_persistence_behavior(self):
        self.call_log = ["init_persistence"]

    def _initialize_scoring_behavior(self):
        self.call_log.append("init_scoring")

    def _initialize_context_behavior(self):
        self.call_log.append("init_context")

    def __call__(self, *args, **kwargs):
        return self.run_declared_execution(*args, **kwargs)

    def resolve_context(self, **context):
        self.call_log.append(f"resolve_context:{sorted(context.keys())}")
        return context

    def score(self, *args, **kwargs):
        _ = (args, kwargs)
        self.call_log.append("score")
        return {"score": 1}

    def save(self):
        self.call_log.append("save")
        return "saved"


@dataclass(eq=False, kw_only=True)
class _StubFrameworkModelDefenseConfig(
    _DefenseContractStubBase,
    FrameworkModelDefenseConfig,
):
    def _validate_model_defense_contract(self):
        self.call_log.append("validate")

    def apply_to(self, estimator=None, data=None):
        _ = (estimator, data)
        self.call_log.append("apply_to")
        return estimator


@dataclass(eq=False, kw_only=True)
class _StubFrameworkDataSamplerContract(FrameworkDataSamplerContract):
    def __call__(self, config):
        _ = config
        return ("train_idx", "test_idx", "val_idx")


@pytest.mark.parametrize(
    "cfg_cls,expected_execution_steps",
    [
        (
            _StubFrameworkDataConfig,
            [
                "load_defaults",
                "load",
                "load_cached",
                "resolve_context",
                "load_data",
                "sample_data",
                "fit_presample",
                "fit_X",
                "fit_y",
                "fit_Xy",
                "score",
                "save",
            ],
        ),
        (
            _StubFrameworkDataPipelineConfig,
            [
                "load_defaults",
                "load",
                "load_cached",
                "resolve_context",
                "load_data",
                "sample_data",
                "fit_presample",
                "fit_X",
                "fit_y",
                "fit_Xy",
                "build_pipeline",
                "run_pipeline",
                "score",
                "save",
            ],
        ),
        (
            _StubFrameworkModelConfig,
            [
                "load_defaults",
                "load",
                "load_cached",
                "resolve_context",
                "init_model",
                "fit_model",
                "score",
                "save",
            ],
        ),
        (
            _StubFrameworkAttackConfig,
            [
                "load_defaults",
                "load",
                "load_cached",
                "resolve_context",
                "build_attack",
                "score",
                "save",
            ],
        ),
        (
            _StubFrameworkDetectorConfig,
            [
                "load_defaults",
                "load",
                "load_cached",
                "resolve_context",
                "build_detector",
                "score",
                "save",
            ],
        ),
        (
            _StubFrameworkExperimentConfig,
            [
                "load_defaults",
                "load",
                "load_cached",
                "resolve_context",
                "run_experiment",
                "score",
                "save",
            ],
        ),
        (
            _StubFrameworkScorerConfig,
            [
                "load_defaults",
                "load",
                "load_cached",
                "resolve_context",
                "score",
                "save",
            ],
        ),
    ],
)
def test_framework_contracts_declare_ordered_lifecycle(
    cfg_cls,
    expected_execution_steps,
):
    cfg = cfg_cls()

    assert cfg.call_log == [
        "init_loading",
        "init_persistence",
        "init_scoring",
        "init_context",
        "validate",
    ]

    results = cfg(context_name="demo")

    assert list(results.keys()) == expected_execution_steps
    expected_call_steps = expected_execution_steps.copy()
    expected_call_steps[3] = "resolve_context:['context_name']"
    assert cfg.call_log[-len(expected_call_steps) :] == expected_call_steps
    assert issubclass(cfg_cls, DeclarativeConfigContract)


def test_framework_model_defense_contract_declares_ordered_lifecycle():
    cfg = _StubFrameworkModelDefenseConfig()

    assert cfg.call_log == [
        "init_persistence",
        "init_scoring",
        "init_context",
        "validate",
    ]

    results = cfg(context_name="demo")
    assert list(results.keys()) == ["resolve_context", "apply_to", "score", "save"]
    assert cfg.call_log[-4:] == [
        "resolve_context:['context_name']",
        "apply_to",
        "score",
        "save",
    ]


def test_framework_data_sampler_contract_callable_shape():
    sampler = _StubFrameworkDataSamplerContract()
    train_idx, test_idx, val_idx = sampler(config=None)
    assert (train_idx, test_idx, val_idx) == ("train_idx", "test_idx", "val_idx")


@dataclass(eq=False, kw_only=True)
class _DuplicateLifecycleContract(DeclarativeConfigContract):
    @classmethod
    def execution_steps(cls):
        return ("run", "run")

    def __call__(self):
        return self.run_declared_execution()

    def run(self):
        return "ok"


def test_contract_rejects_duplicate_execution_steps():
    with pytest.raises(ValueError, match="Duplicate lifecycle step declarations"):
        _DuplicateLifecycleContract()()


# ---------------------------------------------------------------------------
# Adapter boundary enforcement
# ---------------------------------------------------------------------------

_ADAPTER_MIXIN_NAMES = {
    "BaseContractMixin",
    "DataContractMixin",
    "DataPipelineContractMixin",
    "ModelContractMixin",
    "ModelDefenseContractMixin",
    "AttackContractMixin",
    "DetectorContractMixin",
    "ExperimentContractMixin",
    "ScorerContractMixin",
}


def _collect_private_attr_accesses(tree: ast.AST, class_names: set) -> list[str]:
    """Return ``ClassName.method: self._attr`` strings for each violation found."""
    violations: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name in class_names):
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(item):
                # Direct attribute access: self._attr
                if (
                    isinstance(sub, ast.Attribute)
                    and sub.attr.startswith("_")
                    and not sub.attr.startswith("__")
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                ):
                    violations.append(f"{node.name}.{item.name}: self.{sub.attr}")
                # getattr(self, "_attr", ...) call
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "getattr"
                    and len(sub.args) >= 2
                    and isinstance(sub.args[0], ast.Name)
                    and sub.args[0].id == "self"
                    and isinstance(sub.args[1], ast.Constant)
                    and isinstance(sub.args[1].value, str)
                    and sub.args[1].value.startswith("_")
                    and not sub.args[1].value.startswith("__")
                ):
                    violations.append(
                        f"{node.name}.{item.name}: getattr(self, '{sub.args[1].value}', ...)",
                    )
    return violations


def test_adapter_mixin_methods_access_no_private_attributes():
    """Adapter mixin methods must not read or write private attributes on self.

    Enforces the Adapter Attribute Contract from the refactor plan:
    adapters MUST only call public (non-underscore-prefixed) APIs on the
    target config object.
    """
    import deckard.frameworks.adapters as _adapters_mod

    source_path = pathlib.Path(inspect.getfile(_adapters_mod))
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    violations = _collect_private_attr_accesses(tree, _ADAPTER_MIXIN_NAMES)
    assert violations == [], (
        "Adapter mixin methods must not access private attributes on self.\n"
        "Each entry below violates the Adapter Attribute Contract:\n"
        + "\n".join(f"  {v}" for v in violations)
    )
