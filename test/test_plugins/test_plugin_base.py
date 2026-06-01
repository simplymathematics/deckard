from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from deckard.plugins import HookPlugin
from deckard.plugins.base import (
    HookBundle,
    OrchestratorBase,
    RuntimeBase,
    compose_hook_plugins,
)


def test_compose_hook_plugins_flattens_and_deduplicates() -> None:
    first = HookPlugin(hook_name="before_score", method_name="alpha")
    duplicate = HookPlugin(hook_name="before_score", method_name="alpha")
    second = HookPlugin(hook_name="after_score", method_name="beta")
    bundle = HookBundle(name="bundle", hooks=(first, second))

    result = compose_hook_plugins(bundle, [duplicate], None)

    assert [(plugin.hook_name, plugin.method_name) for plugin in result] == [
        ("before_score", "alpha"),
        ("after_score", "beta"),
    ]
    assert result[0] is not first


def test_compose_hook_plugins_rejects_invalid_items() -> None:
    with pytest.raises(TypeError, match="compose_hook_plugins accepts"):
        compose_hook_plugins(["invalid"])


@dataclass(eq=False, kw_only=True)
class _RuntimeHarness(RuntimeBase):
    plugins: list = field(default_factory=list)
    _plugin_objects: list | None = None


class _PluginWithHook:
    def before_score(self, runtime, **kwargs):
        return (runtime, kwargs)


class _PluginWithoutHook:
    pass


def test_runtime_base_caches_instantiated_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RuntimeHarness(plugins=[{"name": "plugin-a"}, {"name": "plugin-b"}])
    seen: list[object] = []

    def _instantiate(spec):
        seen.append(spec)
        return spec

    monkeypatch.setattr(runtime, "_instantiate_plugin", _instantiate)

    first = runtime._get_plugins()
    second = runtime._get_plugins()

    assert first == runtime.plugins
    assert second == first
    assert seen == runtime.plugins


def test_runtime_base_run_plugin_hook_only_calls_matching_hooks() -> None:
    runtime = _RuntimeHarness(
        _plugin_objects=[_PluginWithHook(), _PluginWithoutHook()]
    )

    outputs = runtime._run_plugin_hook("before_score", flag=True)

    assert len(outputs) == 1
    assert outputs[0][1] == {"flag": True}


@dataclass(eq=False, kw_only=True)
class _ScorerCfg:
    stage: object = None


@dataclass(eq=False, kw_only=True)
class _OrchestratorHarness(OrchestratorBase):
    score_mode: str = "test"
    scorer: object = None
    score_dict: dict | None = None
    plugins: list = field(default_factory=list)
    _plugin_objects: list | None = field(default_factory=list)
    calls: list = field(default_factory=list)

    def _run_plugin_hook(self, hook_name: str, **kwargs):
        self.calls.append((hook_name, kwargs))
        payload = kwargs.get("scores")
        if hook_name == "after_score":
            return [{"plugin_metric": 2.0}]
        return [{"hook": hook_name, "payload": payload}] if payload is not None else []

    def score(self, **kwargs):
        self.calls.append(("score", kwargs))
        return {"base_metric": 1.0}


@dataclass(eq=False, kw_only=True)
class _PlainOrchestratorHarness(OrchestratorBase):
    scorer: object = None
    score_dict: dict | None = None
    plugins: list = field(default_factory=list)
    _plugin_objects: list | None = field(default_factory=list)

    def _run_plugin_hook(self, hook_name: str, **kwargs):
        _ = (hook_name, kwargs)
        return []

    @property
    def score(self):
        class _CallableNoSignature:
            __signature__ = None

            def __call__(self, **kwargs):
                return kwargs

        return _CallableNoSignature()


def test_stage_hook_token_and_expand_stage_aliases() -> None:
    runtime = _OrchestratorHarness(
        scorer=type(
            "_Scorer",
            (),
            {
                "configured_scorers": {
                    "a": _ScorerCfg(stage="all"),
                    "b": _ScorerCfg(stage="auto"),
                }
            },
        )(),
    )

    assert runtime._stage_hook_token("before_load") == "pre_load"
    assert runtime._expand_canonical_score_stages(["all", "auto"]) == [
        "pre-load",
        "pre-sample",
        "post-sample",
        "post-pipeline",
    ]


def test_stage_hook_token_and_expand_stage_aliases_reject_invalid_values() -> None:
    runtime = _OrchestratorHarness()

    with pytest.raises(ValueError, match="Unknown score hook stage"):
        runtime._stage_hook_token("definitely-unknown")

    with pytest.raises(ValueError, match="Unsupported score stage"):
        runtime._expand_canonical_score_stages(["definitely-unknown"])


def test_configure_score_orchestration_plugins_populates_hook_plugins() -> None:
    runtime = _OrchestratorHarness(
        scorer=type(
            "_Scorer",
            (),
            {
                "configured_scorers": {
                    "a": _ScorerCfg(stage=["pre-sample", "post-pipeline"])
                }
            },
        )(),
    )

    runtime._configure_score_orchestration_plugins()

    hook_names = [plugin.hook_name for plugin in runtime._plugin_objects]
    assert hook_names == ["before_sample", "after_pipeline"]


def test_iter_configured_score_stages_defaults_when_scorers_missing() -> None:
    runtime = _OrchestratorHarness(scorer=object())

    assert runtime._iter_configured_score_stages() == [runtime.default_stage]


def test_iter_configured_score_stages_flattens_mixed_stage_values() -> None:
    runtime = _OrchestratorHarness(
        scorer=type(
            "_Scorer",
            (),
            {
                "configured_scorers": {
                    "a": _ScorerCfg(stage=None),
                    "b": _ScorerCfg(stage="pre-sample"),
                    "c": _ScorerCfg(stage=["post-sample", "post-pipeline"]),
                },
            },
        )(),
    )

    assert runtime._iter_configured_score_stages() == [
        runtime.default_stage,
        "pre-sample",
        "post-sample",
        "post-pipeline",
    ]


def test_configure_score_orchestration_plugins_skips_unknown_stage_hooks() -> None:
    runtime = _OrchestratorHarness(
        scorer=type(
            "_Scorer",
            (),
            {
                "configured_scorers": {"a": _ScorerCfg(stage=[])},
                "__len__": lambda self: 1,
            },
        )(),
    )
    runtime.score_stage_to_hook = {}

    runtime._configure_score_orchestration_plugins()

    assert runtime._plugin_objects == []


def test_score_orchestration_hook_merges_plugin_scores_and_updates_score_dict() -> (
    None
):
    runtime = _OrchestratorHarness(score_dict={"existing": {"x": 1}})

    result = runtime._score_orchestration_hook(
        "post-pipeline", score_kwargs={"mode": "ignored"}
    )

    assert result["base_metric"] == 1.0
    assert result["plugin_metric"] == 2.0
    assert runtime.score_dict["base_metric"] == 1.0
    assert runtime.score_dict["plugin_metric"] == 2.0


def test_score_orchestration_hook_inactive_returns_none() -> None:
    runtime = _OrchestratorHarness()
    runtime._score_orchestration_active = False

    assert runtime._score_orchestration_hook("post-pipeline") is None


def test_score_orchestration_hook_signature_fallback_adds_mode_and_stage() -> None:
    runtime = _PlainOrchestratorHarness()

    result = runtime._score_orchestration_hook("post-pipeline")

    assert result["mode"] == "test"
    assert result["stage"] == "post-pipeline"


def test_score_orchestration_hook_merges_nested_dict_values() -> None:
    runtime = _OrchestratorHarness(score_dict={"nested": {"old": 1}})

    def _score(**kwargs):
        _ = kwargs
        return {"nested": {"new": 2}}

    runtime.score = _score
    runtime._run_plugin_hook = lambda *args, **kwargs: []

    result = runtime._score_orchestration_hook("post-pipeline")

    assert result == {"nested": {"new": 2}}
    assert runtime.score_dict["nested"] == {"old": 1, "new": 2}


def test_score_orchestration_hook_requires_callable_score_method() -> None:
    runtime = _OrchestratorHarness()
    runtime.score = None

    with pytest.raises(AttributeError, match="has no callable 'score' method"):
        runtime._score_orchestration_hook("post-pipeline")


def test_score_orchestration_hook_returns_none_for_invalid_stage() -> None:
    runtime = _OrchestratorHarness()

    assert runtime._score_orchestration_hook("not-a-stage") is None


def test_run_score_stage_hooks_rejects_invalid_event() -> None:
    runtime = _OrchestratorHarness()

    with pytest.raises(
        ValueError, match="Score hook event must be 'before' or 'after'"
    ):
        runtime._run_score_stage_hooks("during", "post-pipeline")


def test_run_score_stage_hooks_calls_stage_specific_and_generic_hooks() -> None:
    runtime = _OrchestratorHarness()

    runtime._run_score_stage_hooks("before", "post-pipeline", score_kwargs={"a": 1})

    assert runtime.calls[0][0] == "before_score_post_pipeline"
    assert runtime.calls[1][0] == "before_score"
