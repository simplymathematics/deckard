from __future__ import annotations

from dataclasses import dataclass

import pytest

from deckard.plugins.defense import DefenseTypePlugin
from deckard.utils import BaseConfig


@dataclass(eq=False)
class _Runtime(BaseConfig):
    pass


class _ExternalMixin:
    def __init__(self, runtime: BaseConfig):
        self.runtime = runtime

    def __call__(self, *args, **kwargs):
        return self.runtime, (args, kwargs)


@dataclass(eq=False)
class _RuntimeCallable(_Runtime):
    def __call__(self, *args, **kwargs):
        return None, {"args": args, "kwargs": kwargs}


def test_plugin_match_is_case_insensitive_and_subtype_aware() -> None:
    plugin = DefenseTypePlugin(
        mixin_type=_ExternalMixin,
        defense_type="preprocessor",
        defense_subtype="clean",
        excluded_subtypes=("blocked",),
    )

    assert plugin._matches(defense_type="PreProcessor", defense_subtype="CLEAN")
    assert not plugin._matches(defense_type="trainer", defense_subtype="clean")
    assert not plugin._matches(defense_type="preprocessor", defense_subtype="blocked")


def test_resolve_defense_mixins_returns_mixin_when_matched() -> None:
    plugin = DefenseTypePlugin(
        mixin_type=_ExternalMixin,
        defense_type="trainer",
    )
    runtime = _Runtime()

    mixins = plugin.resolve_defense_mixins(
        runtime,
        defense_type="trainer",
        defense_subtype=None,
        default_mixins=(),
    )

    assert mixins == (_ExternalMixin,)


def test_resolve_mixin_type_string_uses_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = DefenseTypePlugin(
        mixin_type="pkg.ExternalMixin",
        defense_type="trainer",
    )
    monkeypatch.setattr("deckard.utils.resolve_class", lambda _: _ExternalMixin)

    resolved = plugin._resolve_mixin_type()

    assert resolved is _ExternalMixin
    assert plugin.mixin_type is _ExternalMixin


def test_resolve_defense_handler_returns_callable_only_when_matched() -> None:
    plugin = DefenseTypePlugin(mixin_type=_ExternalMixin, defense_type="detector")
    runtime = _Runtime()

    matched = plugin.resolve_defense_handler(
        runtime,
        defense_type="detector",
        defense_subtype=None,
        default_handler=None,
        default_mixins=(),
    )
    unmatched = plugin.resolve_defense_handler(
        runtime,
        defense_type="trainer",
        defense_subtype=None,
        default_handler=None,
        default_mixins=(),
    )

    assert callable(matched)
    assert unmatched is None


def test_plugin_call_uses_runtime_directly_when_mixin_in_mro() -> None:
    runtime = _RuntimeCallable()
    plugin = DefenseTypePlugin(
        mixin_type=_RuntimeCallable,
        defense_type="trainer",
    )

    result = plugin(runtime, 1, step="fit")

    assert result[0] is None
    assert result[1]["args"] == (1,)
    assert result[1]["kwargs"] == {"step": "fit"}


def test_plugin_call_wraps_runtime_when_mixin_is_external() -> None:
    runtime = _Runtime()
    plugin = DefenseTypePlugin(
        mixin_type=_ExternalMixin,
        defense_type="trainer",
    )

    result = plugin(runtime, "payload", stage="score")

    assert result[0] is runtime
    assert result[1][0] == ("payload",)
    assert result[1][1] == {"stage": "score"}
