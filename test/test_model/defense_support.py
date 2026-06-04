"""Shared helpers for model defense unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class KwargClassifierDefense:
    def __init__(self, classifier=None, **kwargs):
        self.classifier = classifier
        self.kwargs = kwargs

    def get_classifier(self):
        return {"wrapped": self.classifier, "kwargs": self.kwargs}


class PositionalClassifierDefense:
    def __init__(self, classifier, **kwargs):
        self.classifier = classifier
        self.kwargs = kwargs


def make_defense_config(config_cls: type[Any], *, defense_params: dict[str, Any]):
    cfg = config_cls()
    cfg.defense_params = defense_params
    cfg._model = None
    return cfg


def patch_torch_model_check(monkeypatch, target: str) -> None:
    monkeypatch.setattr(target, lambda _model: True)


def make_defense_call_kwargs(
    *,
    defense_type: str,
    defense_subtype: str,
    defense_class: type[Any],
    art_class: Any = object,
    init_params: dict[str, Any] | None = None,
    base_estimator: Any | None = None,
    existing_preprocessors: list[Any] | None = None,
    existing_postprocessors: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "data": None,
        "defense_type": defense_type,
        "defense_subtype": defense_subtype,
        "defense_class": defense_class,
        "art_class": art_class,
        "init_params": init_params or {},
        "base_estimator": (
            base_estimator if base_estimator is not None else SimpleNamespace()
        ),
        "existing_preprocessors": existing_preprocessors or [],
        "existing_postprocessors": existing_postprocessors or [],
    }


def setup_wrapped_art_builder(cfg: Any, monkeypatch, target: str) -> Any:
    wrapped = SimpleNamespace(name="wrapped")
    cfg._build_art_wrapper = lambda **kwargs: wrapped
    patch_torch_model_check(monkeypatch, target)
    return wrapped


def assert_rejects_non_torch_estimator(
    cfg: Any,
    *,
    defense_type: str,
    defense_subtype: str,
    defense_class: type[Any],
) -> None:
    with pytest.raises(ValueError, match="only support neural-network models"):
        cfg(
            **make_defense_call_kwargs(
                defense_type=defense_type,
                defense_subtype=defense_subtype,
                defense_class=defense_class,
                base_estimator=object(),
            ),
        )


def run_wrapped_defense(
    cfg: Any,
    monkeypatch,
    *,
    torch_model_check_target: str,
    defense_type: str,
    defense_subtype: str,
    defense_class: type[Any],
    init_params: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    wrapped = setup_wrapped_art_builder(
        cfg,
        monkeypatch,
        target=torch_model_check_target,
    )
    defense, defended_estimator = cfg(
        **make_defense_call_kwargs(
            defense_type=defense_type,
            defense_subtype=defense_subtype,
            defense_class=defense_class,
            init_params=init_params,
        ),
    )
    return wrapped, defense, defended_estimator
