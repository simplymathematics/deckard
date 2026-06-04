from __future__ import annotations

from pathlib import Path

import pytest

from deckard.data.base import DataConfig
from deckard.data.canon import CANONICAL_DATA_TIMES
from deckard.plugins import HookPlugin


def _core_config() -> DataConfig:
    return DataConfig(
        name="make_classification",
        data_params={
            "n_samples": 48,
            "n_features": 8,
            "n_informative": 4,
            "n_redundant": 0,
            "random_state": 7,
        },
        classifier=True,
        scorer=lambda **kwargs: {"core_rows": len(kwargs.get("y", []))},
    )


def _fairlearn_config():
    pytest.importorskip("fairlearn")
    from deckard.plugins.fairlearn import FairlearnDataConfig

    return FairlearnDataConfig(
        name="make_classification",
        data_params={
            "n_samples": 48,
            "n_features": 8,
            "n_informative": 4,
            "n_redundant": 0,
            "random_state": 11,
        },
        classifier=True,
        sensitive_columns=["feature_0"],
        fairness_defense=False,
        scorer=lambda **kwargs: {"fair_rows": len(kwargs.get("y", []))},
    )


def _anjana_config():
    from deckard.plugins.anjana import AnjanaDataConfig

    return AnjanaDataConfig(
        name="make_classification",
        data_params={
            "n_samples": 48,
            "n_features": 8,
            "n_informative": 4,
            "n_redundant": 0,
            "random_state": 19,
        },
        classifier=True,
        anjana_defense=False,
        quasi_identifiers=["feature_0", "feature_1"],
        sensitive_attribute="target",
        sensitive_columns=["feature_0"],
        scorer=lambda **kwargs: {"k_anonymity": float(len(kwargs.get("X", [])) > -1)},
    )


def _pytorch_config():
    torch = pytest.importorskip("torch")
    from deckard.frameworks.pytorch.data import PytorchDataConfig

    X = torch.randn(48, 8)
    y = torch.randint(0, 2, (48,))
    return PytorchDataConfig(
        name="torch.utils.data.TensorDataset",
        data_params={"_args_": [X, y]},
        classifier=True,
        sampler={
            "train_size": 12,
            "test_size": 8,
            "random_state": 42,
            "name": "split",
        },
        scorer=lambda **kwargs: {"torch_rows": len(kwargs.get("y", []))},
    )


@pytest.mark.parametrize(
    "builder",
    [_core_config, _pytorch_config, _fairlearn_config, _anjana_config],
)
def test_cross_family_data_runtime_contract(builder, tmp_path: Path):
    cfg = builder()

    for method_name in (
        "load_dataset",
        "fit_transform",
        "sample",
        "score",
        "__call__",
    ):
        assert callable(getattr(cfg, method_name, None)), method_name

    for time_key in CANONICAL_DATA_TIMES:
        assert time_key in cfg.times

    with pytest.raises(ValueError):
        cfg.score(mode="post-pipeline")

    score_file = tmp_path / f"{type(cfg).__name__}_scores.json"
    scores = cfg(files={"score_file": str(score_file)})

    assert isinstance(scores, dict)
    assert isinstance(cfg.files, dict)
    assert "score_file" in cfg.files
    assert "data_load_time" in scores
    assert "data_sample_time" in scores


def test_core_and_framework_score_hooks_use_canonical_stage_names(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from deckard.frameworks.pytorch.data import PytorchDataConfig

    core = _core_config()

    X = torch.randn(48, 8)
    y = torch.randint(0, 2, (48,))
    framework = PytorchDataConfig(
        name="torch.utils.data.TensorDataset",
        data_params={"_args_": [X, y]},
        classifier=True,
        sampler={"name": "split", "test_size": 0.2, "random_state": 42},
        scorer=lambda **kwargs: {"torch_rows": len(kwargs.get("y", []))},
    )

    def attach_stage_recorder(cfg, prefix: str):
        events = []
        cfg.plugins = [
            HookPlugin(
                hook_name="before_score_post_pipeline",
                method_name="_record_before_score",
            ),
            HookPlugin(
                hook_name="after_score_post_pipeline",
                method_name="_record_after_score",
            ),
        ]
        cfg._plugin_objects = None
        cfg._record_before_score = lambda **kwargs: events.append(
            f"{prefix}:before:{kwargs.get('stage')}",
        )
        cfg._record_after_score = lambda **kwargs: events.append(
            f"{prefix}:after:{kwargs.get('stage')}",
        )
        cfg._initialize_runtime_components()
        return events

    core_events = attach_stage_recorder(core, "core")
    framework_events = attach_stage_recorder(framework, "framework")

    core(files={"score_file": str(tmp_path / "core_scores.json")})
    framework(files={"score_file": str(tmp_path / "framework_scores.json")})

    assert "core:before:post-pipeline" in core_events
    assert "core:after:post-pipeline" in core_events
    assert "framework:before:post-pipeline" in framework_events
    assert "framework:after:post-pipeline" in framework_events
