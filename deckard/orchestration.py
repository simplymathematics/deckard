"""Core orchestration primitives for score-stage routing and runtime hook order.

This module owns canonical stage/mode normalization and score-hook dispatch.
It does not serialize artifacts or mutate config payload schemas.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Mapping

from .artifacts import ScoreDict
from .plugins.base import OrchestratorBase, RuntimeBase

CANONICAL_RUNTIME_METHODS: Final[tuple[str, ...]] = (
    "load",
    "sample",
    "pipeline",
)

CANONICAL_SCORE_STAGES: Final[tuple[str, ...]] = (
    "pre-load",
    "pre-sample",
    "post-sample",
    "post-pipeline",
    "all",
    "auto",
)

CANONICAL_SCORE_MODES: Final[tuple[str, ...]] = (
    "train",
    "test",
    "val",
    "all",
)

DEFAULT_SCORE_STAGE: Final[str] = "post-pipeline"
DEFAULT_SCORE_MODE: Final[str] = "test"

STAGE_ALIASES: Final[dict[str, str]] = {
    "pre-load": "pre-load",
    "preload": "pre-load",
    "before-load": "pre-load",
    "before_load": "pre-load",
    "pre-sample": "pre-sample",
    "pre_sample": "pre-sample",
    "presample": "pre-sample",
    "post-sample": "post-sample",
    "post_sample": "post-sample",
    "postsample": "post-sample",
    "post-pipeline": "post-pipeline",
    "post_pipeline": "post-pipeline",
    "postpipeline": "post-pipeline",
}

MODE_ALIASES: Final[dict[str, str]] = {
    "train": "train",
    "training": "train",
    "test": "test",
    "eval": "test",
    "evaluation": "test",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "all": "all",
}

EVENT_ALIASES: Final[dict[str, str]] = {
    "pre": "before",
    "before": "before",
    "post": "after",
    "after": "after",
}

RUNTIME_SPLIT_ALIASES: Final[dict[str, str]] = {
    "train": "train",
    "test": "test",
    "val": "val",
    "all": "all",
    "attack": "test",
    "attack-val": "val",
    "auto": "test",
    "benign": "test",
    "adversarial": "test",
}

_SPLIT_DATA_ATTRS: Final[dict[str, tuple[str, str]]] = {
    "train": ("y_train", "X_train"),
    "test": ("y_test", "X_test"),
    "val": ("y_val", "X_val"),
    "all": ("_y", "_X"),
}

_SPLIT_SENSITIVE_ATTRS: Final[dict[str, str]] = {
    "train": "_sensitive_train",
    "test": "_sensitive_test",
    "val": "_sensitive_val",
    "all": "_sensitive_all",
}


def stage_hook_token(stage: str) -> str:
    """Convert canonical score stage names into hook-safe tokens."""
    key = str(stage).strip().lower().replace(" ", "-")
    if key in STAGE_ALIASES:
        return STAGE_ALIASES[key].replace("-", "_")
    raise ValueError(
        f"Unknown score hook stage '{stage}'. Must be one of {list(CANONICAL_SCORE_STAGES)}",
    )


def normalize_score_mode(mode: str) -> str:
    """Normalize score split mode names to canonical tokens."""
    key = str(mode).strip().lower().replace(" ", "-")
    if key in MODE_ALIASES:
        return MODE_ALIASES[key]
    raise ValueError(
        f"Unknown score mode '{mode}'. Must be one of {list(CANONICAL_SCORE_MODES)}",
    )


def normalize_runtime_split_mode(
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
    default: str = "test",
) -> str:
    """Normalize stage/split aliases to canonical runtime split tokens."""
    token = str(mode or default).strip().lower()
    alias_map = dict(RUNTIME_SPLIT_ALIASES)
    if aliases:
        alias_map.update(
            {
                str(key).strip().lower(): str(value).strip().lower()
                for key, value in aliases.items()
            },
        )
    resolved = alias_map.get(token, token)
    if resolved not in _SPLIT_DATA_ATTRS:
        raise ValueError(
            f"Unknown runtime split mode '{mode}'. Expected one of {sorted(_SPLIT_DATA_ATTRS)}",
        )
    return resolved


def resolve_data_split_payload(
    data: Any,
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
    fallback_to_all: bool = False,
) -> tuple[Any, Any]:
    """Resolve ``(y, X)`` payload for a runtime split mode from a data object."""
    if data is None:
        return None, None
    resolved_mode = normalize_runtime_split_mode(mode, aliases=aliases)
    y_attr, x_attr = _SPLIT_DATA_ATTRS[resolved_mode]
    y = getattr(data, y_attr, None)
    X = getattr(data, x_attr, None)
    if fallback_to_all and resolved_mode != "all":
        if y is None:
            y = getattr(data, "_y", None)
        if X is None:
            X = getattr(data, "_X", None)
    return y, X


def resolve_sensitive_split_payload(
    data: Any,
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
    fallback_to_all: bool = False,
) -> Any:
    """Resolve sensitive-feature payload for a runtime split mode."""
    if data is None:
        return None
    resolved_mode = normalize_runtime_split_mode(mode, aliases=aliases)
    sensitive_attr = _SPLIT_SENSITIVE_ATTRS[resolved_mode]
    sensitive = getattr(data, sensitive_attr, None)
    if sensitive is None:
        legacy_attr = sensitive_attr.removeprefix("_")
        sensitive = getattr(data, legacy_attr, None)
    if sensitive is None and fallback_to_all and resolved_mode != "all":
        all_attr = _SPLIT_SENSITIVE_ATTRS["all"]
        sensitive = getattr(data, all_attr, None)
        if sensitive is None:
            sensitive = getattr(data, all_attr.removeprefix("_"), None)
    return sensitive


@dataclass(eq=False, kw_only=True)
class DataRuntimeStateMixin(RuntimeBase):
    """Reusable runtime-state copy behavior for data-like components.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def _copy_runtime_state_to(self, target: Any) -> None:
        runtime_fields = [
            "score_dict",
            "times",
            "files",
            "data_load_time",
            "data_sample_time",
            "data_pipeline_time",
            "data_score_time",
            "_X",
            "_y",
            "train_indices",
            "test_indices",
            "val_indices",
            "X_train",
            "y_train",
            "X_test",
            "y_test",
            "X_val",
            "y_val",
            "train_n",
            "test_n",
            "val_n",
            "pipeline_fit_n",
            "pipeline_transform_n",
            "pipeline_fit_time",
            "pipeline_transform_time",
            "pipeline_y_fit_n",
            "pipeline_y_fit_time",
            "pipeline_y_transform_n",
            "pipeline_y_transform_time",
        ]
        for attr in runtime_fields:
            if hasattr(self, attr):
                setattr(target, attr, getattr(self, attr, None))


@dataclass(eq=False, kw_only=True)
class ScoreOrchestratorMixin(OrchestratorBase, DataRuntimeStateMixin):
    """Shared score-stage orchestration for data-like runtimes.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    default_stage: Final[str] = DEFAULT_SCORE_STAGE
    stage_aliases: ClassVar[dict[str, str]] = STAGE_ALIASES
    mode_aliases: ClassVar[dict[str, str]] = MODE_ALIASES
    score_stage_aliases: ClassVar[dict[str, str]] = STAGE_ALIASES
    score_stage_order: ClassVar[tuple[str, ...]] = tuple(
        stage for stage in CANONICAL_SCORE_STAGES if stage not in {"all", "auto"}
    )
    score_event_aliases: ClassVar[dict[str, str]] = EVENT_ALIASES
    score_stage_to_hook: ClassVar[dict[str, str]] = {
        "pre-load": "before_load_data",
        "pre-sample": "before_sample",
        "post-sample": "after_sample",
        "post-pipeline": "after_pipeline",
    }
    _score_orchestration_active: bool = field(default=True, init=False, metadata={'help': 'Configuration field: _score_orchestration_active.'}, repr=False)

    def _normalize_score_mode(self, mode: str) -> str:
        return normalize_score_mode(mode)

    def _stage_hook_token(self, stage: str) -> str:
        return stage_hook_token(stage)

    def _iter_configured_score_stages(self) -> list[str]:
        scorer = getattr(self, "scorer", None)
        configured = getattr(scorer, "configured_scorers", None)
        if not isinstance(configured, dict) or len(configured) == 0:
            return [self.default_stage]

        raw_stages: list[str] = []
        for scorer_cfg in configured.values():
            stage_value = getattr(scorer_cfg, "stage", None)
            if stage_value in [None, "", []]:
                raw_stages.append(self.default_stage)
                continue
            if isinstance(stage_value, str):
                raw_stages.append(stage_value)
                continue
            raw_stages.extend([str(token) for token in stage_value])

        if len(raw_stages) == 0:
            return [self.default_stage]
        return raw_stages

    def _expand_canonical_score_stages(self, raw_stages: list[str]) -> list[str]:
        ordered = list(self.score_stage_order)
        expanded: list[str] = []

        for token in raw_stages:
            normalized = str(token).strip().lower().replace("_", "-")
            if normalized in {"", "auto"}:
                expanded.append(self.default_stage)
                continue
            if normalized == "all":
                expanded.extend(ordered)
                continue
            if normalized in ordered:
                expanded.append(normalized)
                continue
            raise ValueError(
                f"Unsupported score stage '{token}'. Expected one of {ordered + ['all', 'auto']}",
            )

        deduped: list[str] = []
        for stage in ordered:
            if stage in expanded and stage not in deduped:
                deduped.append(stage)
        return deduped or [self.default_stage]

    def _configure_score_orchestration_plugins(self) -> None:
        from .plugins import HookPlugin

        stage_tokens = self._expand_canonical_score_stages(
            self._iter_configured_score_stages(),
        )
        score_plugins = [
            HookPlugin(
                hook_name=self.score_stage_to_hook[stage],
                method_name="_score_orchestration_hook",
                method_kwargs={"stage": stage},
            )
            for stage in stage_tokens
            if stage in self.score_stage_to_hook
        ]
        if len(score_plugins) == 0:
            return
        if not hasattr(self, "_plugin_objects") or self._plugin_objects is None:
            self._plugin_objects = []
        self._plugin_objects.extend(score_plugins)

    def _score_orchestration_hook(self, stage: str, **kwargs: Any):
        if not self._score_orchestration_active:
            return None
        try:
            self._stage_hook_token(stage)
        except ValueError:
            return None
        mode = kwargs.pop("mode", None)
        mode = self._normalize_score_mode(mode or getattr(self, "score_mode", "test"))
        score_kwargs = kwargs.pop("score_kwargs", None) or {}
        if not isinstance(score_kwargs, dict):
            score_kwargs = dict(score_kwargs)
        score_kwargs.pop("mode", None)
        score_kwargs.pop("stage", None)
        self._run_score_stage_hooks("before", stage, score_kwargs=score_kwargs)
        score_fn = getattr(self, "score", None)
        if not callable(score_fn):
            raise AttributeError(
                f"{type(self).__name__} has no callable 'score' method",
            )
        score_call_kwargs = dict(score_kwargs)
        try:
            signature = inspect.signature(score_fn)
        except (TypeError, ValueError):
            signature = None
        if signature is None:
            score_call_kwargs.setdefault("mode", mode)
            score_call_kwargs.setdefault("stage", stage)
        else:
            params = signature.parameters
            has_var_kw = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in params.values()
            )
            if "mode" in params or has_var_kw:
                score_call_kwargs.setdefault("mode", mode)
            if "stage" in params or has_var_kw:
                score_call_kwargs.setdefault("stage", stage)
        result = score_fn(**score_call_kwargs)
        plugin_scores = self._run_score_stage_hooks("after", stage, scores=result)
        if isinstance(result, dict):
            for plugin_score in plugin_scores:
                if isinstance(plugin_score, dict):
                    result.update(plugin_score)
            if getattr(self, "score_dict", None) is None:
                self.score_dict = ScoreDict()
            else:
                self.score_dict = ScoreDict.from_payload(self.score_dict)
            for key, value in result.items():
                if (
                    key in self.score_dict
                    and isinstance(self.score_dict.get(key), dict)
                    and isinstance(value, dict)
                ):
                    self.score_dict[key].update(value)
                else:
                    self.score_dict[key] = value
        return result

    def _run_score_stage_hooks(
        self,
        when: str,
        stage: str,
        **kwargs: Any,
    ) -> list[Any]:
        event = str(when).strip().lower()
        if event not in self.score_event_aliases:
            raise ValueError(
                f"Score hook event must be 'before' or 'after', got {when}",
            )
        event = self.score_event_aliases[event]
        stage_token = self._stage_hook_token(stage)
        stage = stage_token.replace("_", "-")
        stage_kwargs = {"stage": stage, **kwargs}
        outputs: list[Any] = []
        outputs.extend(
            self._run_plugin_hook(f"{event}_score_{stage_token}", **stage_kwargs),
        )
        outputs.extend(self._run_plugin_hook(f"{event}_score", **stage_kwargs))
        return outputs


__all__ = [
    "CANONICAL_RUNTIME_METHODS",
    "CANONICAL_SCORE_STAGES",
    "CANONICAL_SCORE_MODES",
    "DEFAULT_SCORE_STAGE",
    "DEFAULT_SCORE_MODE",
    "STAGE_ALIASES",
    "MODE_ALIASES",
    "EVENT_ALIASES",
    "RUNTIME_SPLIT_ALIASES",
    "stage_hook_token",
    "normalize_score_mode",
    "normalize_runtime_split_mode",
    "resolve_data_split_payload",
    "resolve_sensitive_split_payload",
    "DataRuntimeStateMixin",
    "ScoreOrchestratorMixin",
]
