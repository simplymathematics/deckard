"""Core orchestration primitives for score-stage routing and runtime hook order.

This module owns canonical stage/mode normalization and score-hook dispatch.
It does not serialize artifacts or mutate config payload schemas.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Mapping

from .artifacts import ScoreDict
from .plugins import RuntimeBase

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
    "pre-pipeline": "post-sample",
    "prepipeline": "post-sample",
    "pre_pipeline": "post-sample",
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

SCORE_STAGE_TO_HOOK: Final[dict[str, str]] = {
    "pre-load": "before_load_data",
    "pre-sample": "before_sample",
    "post-sample": "after_sample",
    "post-pipeline": "after_pipeline",
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

VALIDATION_SPLIT_RESET_FIELDS: Final[tuple[str, ...]] = (
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
)


def normalize_score_stage(
    stage: str | None,
    *,
    default: str = DEFAULT_SCORE_STAGE,
    aliases: Mapping[str, str] | None = None,
    allow_all_auto: bool = True,
) -> str:
    """Normalize score stage aliases to canonical stage tokens."""
    token = str(stage or default).strip().lower().replace("_", "-").replace(" ", "-")
    if allow_all_auto and token in {"all", "auto"}:
        return token

    alias_map = dict(STAGE_ALIASES)
    if aliases:
        alias_map.update(
            {
                str(key)
                .strip()
                .lower()
                .replace("_", "-"): str(value)
                .strip()
                .lower()
                .replace("_", "-")
                for key, value in aliases.items()
            },
        )

    resolved = alias_map.get(token)
    if resolved is None:
        raise ValueError(
            f"Unknown score hook stage '{stage}'. Must be one of {list(CANONICAL_SCORE_STAGES)}",
        )
    return resolved


def expand_score_stages(
    raw_stages: list[str],
    *,
    default_stage: str = DEFAULT_SCORE_STAGE,
    stage_order: tuple[str, ...] = tuple(
        stage for stage in CANONICAL_SCORE_STAGES if stage not in {"all", "auto"}
    ),
    stage_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    """Expand configured score stages, resolving aliases and all/auto tokens."""
    ordered = list(stage_order)
    expanded: list[str] = []
    aliases = dict(stage_aliases or STAGE_ALIASES)

    for token in raw_stages:
        try:
            normalized = normalize_score_stage(
                token,
                default=default_stage,
                aliases=aliases,
                allow_all_auto=True,
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported score stage '{token}'. Expected one of {ordered + ['all', 'auto']}",
            ) from exc
        if normalized in {"", "auto"}:
            expanded.append(default_stage)
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
    return deduped or [default_stage]


def stage_hook_token(stage: str) -> str:
    """Convert canonical score stage names into hook-safe tokens."""
    return normalize_score_stage(stage, allow_all_auto=False).replace("-", "_")


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
) -> tuple[Any, Any]:
    """Resolve ``(y, X)`` payload for a runtime split mode from a data object."""
    if data is None:
        return None, None
    resolved_mode = normalize_runtime_split_mode(mode, aliases=aliases)
    y_attr, x_attr = _SPLIT_DATA_ATTRS[resolved_mode]
    y = getattr(data, y_attr, None)
    X = getattr(data, x_attr, None)
    return y, X


def resolve_sensitive_split_payload(
    data: Any,
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
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
    return sensitive


def resolve_attack_split_payload(
    data: Any,
    requested_mode: str | None,
    *,
    error_message: str,
    on_fallback: Any = None,
) -> tuple[str, Any, Any]:
    """Resolve attack split payload preferring requested train/val mode, else test."""
    mode_token = normalize_runtime_split_mode(requested_mode, default="test")
    if mode_token in {"train", "val"}:
        y_attr, x_attr = _SPLIT_DATA_ATTRS[mode_token]
        y = getattr(data, y_attr, None)
        X = getattr(data, x_attr, None)
        if X is not None and y is not None:
            return mode_token, X, y
        if callable(on_fallback):
            on_fallback(mode_token)

    y_test = getattr(data, "y_test", None)
    X_test = getattr(data, "X_test", None)
    if X_test is None or y_test is None:
        raise ValueError(error_message)
    return "test", X_test, y_test


def ensure_validation_split_available(
    data: Any,
    *,
    allow_call_fallback: bool = False,
    allow_test_fallback: bool = False,
    error_message: str = "Validation split is not available.",
) -> tuple[Any, Any]:
    """Ensure ``(X_val, y_val)`` are available for validation-scoped runtime work."""
    if data is None:
        raise ValueError(error_message)

    X_val = getattr(data, "X_val", None)
    y_val = getattr(data, "y_val", None)
    if X_val is not None and y_val is not None:
        return X_val, y_val

    can_resample = (
        callable(getattr(data, "sample", None))
        and getattr(data, "_X", None) is not None
        and getattr(data, "_y", None) is not None
    )
    if can_resample:
        data.data_sample_time = None
        for attr in VALIDATION_SPLIT_RESET_FIELDS:
            setattr(data, attr, None)
        data.sample()
        X_val = getattr(data, "X_val", None)
        y_val = getattr(data, "y_val", None)

    if (
        (X_val is None or y_val is None)
        and allow_call_fallback
        and callable(
            getattr(data, "__call__", None),
        )
    ):
        data()
        X_val = getattr(data, "X_val", None)
        y_val = getattr(data, "y_val", None)

    if (X_val is None or y_val is None) and allow_test_fallback:
        X_test = getattr(data, "X_test", None)
        y_test = getattr(data, "y_test", None)
        if X_test is not None and y_test is not None:
            data.X_val = X_test
            data.y_val = y_test
            data.val_n = len(y_test)
            X_val = data.X_val
            y_val = data.y_val

    if X_val is None or y_val is None:
        raise ValueError(error_message)
    return X_val, y_val


def normalize_keyword_filters(
    filters: Mapping[str, Any] | None,
) -> dict[str, tuple[str, ...]]:
    """Normalize include/exclude keyword filter config into canonical tuples."""
    payload = dict(filters or {})

    def _coerce(value: Any) -> tuple[str, ...]:
        if value in [None, "", []]:
            return ()
        if isinstance(value, str):
            tokens = [value]
        elif isinstance(value, (list, tuple, set)):
            tokens = list(value)
        else:
            tokens = [value]
        normalized: list[str] = []
        for token in tokens:
            text = str(token).strip().lower()
            if text == "":
                continue
            normalized.append(text)
        return tuple(normalized)

    return {
        "include": _coerce(payload.get("include")),
        "exclude": _coerce(payload.get("exclude")),
    }


def filter_scores_by_keywords(
    scores: Mapping[str, Any] | None,
    *,
    include_keywords: tuple[str, ...] = (),
    exclude_keywords: tuple[str, ...] = (),
    context_keywords: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Filter score keys by include/exclude keyword rules and context tokens."""
    if not isinstance(scores, Mapping):
        return {}

    include = tuple(str(token).strip().lower() for token in include_keywords if token)
    exclude = tuple(str(token).strip().lower() for token in exclude_keywords if token)
    context = tuple(str(token).strip().lower() for token in context_keywords if token)

    filtered: dict[str, Any] = {}
    for key, value in scores.items():
        key_token = str(key).strip().lower()
        haystack = " ".join((key_token, *context)).strip()

        if len(include) > 0 and not any(token in haystack for token in include):
            continue
        if len(exclude) > 0 and any(token in haystack for token in exclude):
            continue
        filtered[str(key)] = value
    return filtered


@dataclass(eq=False, kw_only=True)
class ScoreOrchestratorMixin(RuntimeBase):
    """Unified runtime + score-stage orchestration mixin for runtimes."""

    default_stage: str = DEFAULT_SCORE_STAGE
    stage_scoring_enabled: bool = False
    stage_score_filters: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Optional include/exclude keyword filters for stage-scoped scoring.",
        },
    )
    score_stage_aliases: ClassVar[dict[str, str]] = dict(STAGE_ALIASES)
    score_stage_order: ClassVar[tuple[str, ...]] = tuple(
        stage for stage in CANONICAL_SCORE_STAGES if stage not in {"all", "auto"}
    )
    score_event_aliases: ClassVar[dict[str, str]] = dict(EVENT_ALIASES)
    score_stage_to_hook: ClassVar[dict[str, str]] = dict(SCORE_STAGE_TO_HOOK)
    _score_orchestration_active: bool = field(
        default=True,
        init=False,
        metadata={"help": "Configuration field: _score_orchestration_active."},
        repr=False,
    )

    def _normalize_score_mode(self, mode: str) -> str:
        return normalize_score_mode(mode)

    def _normalize_stage_score_filters(self) -> None:
        """Normalize keyword filter config for stage-scoped score recording."""
        self.stage_score_filters = dict(
            normalize_keyword_filters(getattr(self, "stage_score_filters", None)),
        )

    def _stage_score_context_keywords(
        self,
        *,
        stage: str,
        mode: str,
        component: str | None = None,
    ) -> tuple[str, ...]:
        """Return canonical context tokens used by keyword-filtered stage scores."""
        cls_name = type(self).__name__.strip().lower()
        if cls_name.endswith("config"):
            cls_name = cls_name[: -len("config")]
        module_domain = str(type(self).__module__).split(".")
        domain = module_domain[1] if len(module_domain) > 1 else cls_name
        return tuple(
            token
            for token in (stage, mode, component, domain, cls_name)
            if token not in [None, ""]
        )

    def _record_stage_scores(
        self,
        *,
        stage: str,
        mode: str,
        scores: Mapping[str, Any] | None,
        component: str | None = None,
    ) -> dict[str, Any]:
        """Persist filtered stage scores under score_dict['stages'][stage][mode]."""
        normalized_stage = self._normalize_score_stage(stage, allow_all_auto=False)
        normalized_mode = self._normalize_score_mode(mode)
        self._normalize_stage_score_filters()

        include_keywords = tuple(self.stage_score_filters.get("include", ()))
        exclude_keywords = tuple(self.stage_score_filters.get("exclude", ()))
        filtered_scores = filter_scores_by_keywords(
            scores,
            include_keywords=include_keywords,
            exclude_keywords=exclude_keywords,
            context_keywords=self._stage_score_context_keywords(
                stage=normalized_stage,
                mode=normalized_mode,
                component=component,
            ),
        )
        if len(filtered_scores) == 0:
            return {}

        if getattr(self, "score_dict", None) is None:
            self.score_dict = ScoreDict()
        else:
            self.score_dict = ScoreDict.from_payload(self.score_dict)

        stages_bucket = self.score_dict.setdefault("stages", {})
        stage_bucket = stages_bucket.setdefault(normalized_stage, {})
        mode_bucket = stage_bucket.setdefault(normalized_mode, {})
        mode_bucket.update(filtered_scores)

        if component not in [None, ""]:
            components_bucket = stage_bucket.setdefault("components", {})
            component_bucket = components_bucket.setdefault(str(component), {})
            component_mode_bucket = component_bucket.setdefault(normalized_mode, {})
            component_mode_bucket.update(filtered_scores)
        return filtered_scores

    def _build_stage_score_payload(
        self,
        data: Any,
        *,
        stage: str,
        mode: str,
        component: str | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any] | None:
        """Build raw score payload for a stage-scoring pass.

        Domain runtimes override this to compute scores from canonical split payloads.
        """
        return {}

    def _run_stage_scoring_pass(
        self,
        data: Any,
        *,
        stage: str,
        component: str | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute and persist stage-scoped scores for the active runtime mode."""
        if not getattr(self, "stage_scoring_enabled", False):
            return {}
        normalized_stage = self._normalize_score_stage(stage, allow_all_auto=False)
        normalized_mode = self._normalize_score_mode(
            mode or getattr(self, "score_mode", "test"),
        )
        payload = self._build_stage_score_payload(
            data,
            stage=normalized_stage,
            mode=normalized_mode,
            component=component,
            **kwargs,
        )
        if not isinstance(payload, Mapping):
            return {}
        return self._record_stage_scores(
            stage=normalized_stage,
            mode=normalized_mode,
            component=component,
            scores=payload,
        )

    def _normalize_score_stage(
        self,
        stage: str | None,
        *,
        allow_all_auto: bool = True,
    ) -> str:
        """Normalize score stage names for this runtime orchestration context."""
        return normalize_score_stage(
            stage,
            default=self.default_stage,
            aliases=self.score_stage_aliases,
            allow_all_auto=allow_all_auto,
        )

    @staticmethod
    def _hook_token(value: str | None) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    def _normalize_hook_event(self, when: str) -> str:
        event = str(when).strip().lower()
        if event not in self.score_event_aliases:
            raise ValueError(
                f"Score hook event must be 'before' or 'after', got {when}",
            )
        return self.score_event_aliases[event]

    def _build_generic_stage_hook_names(
        self,
        *,
        event: str,
        stage: str,
        component: str | None = None,
        previous_component: str | None = None,
        next_component: str | None = None,
    ) -> list[str]:
        stage_token = self._hook_token(stage)
        component_token = self._hook_token(component)
        prev_token = self._hook_token(previous_component)
        next_token = self._hook_token(next_component)

        names: list[str] = [f"{event}_{stage_token}"]
        if component_token:
            names.extend(
                [
                    f"{event}_{component_token}_{stage_token}",
                    f"{event}_{stage_token}_{component_token}",
                ],
            )
        if prev_token and next_token:
            names.extend(
                [
                    f"{event}_between_{prev_token}_{next_token}",
                    f"{event}_after_{prev_token}_before_{next_token}",
                    f"{event}_{prev_token}_before_{next_token}",
                ],
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        return deduped

    def _run_generic_stage_hooks(
        self,
        when: str,
        stage: str,
        *,
        component: str | None = None,
        previous_component: str | None = None,
        next_component: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        event = self._normalize_hook_event(when)
        hook_names = self._build_generic_stage_hook_names(
            event=event,
            stage=stage,
            component=component,
            previous_component=previous_component,
            next_component=next_component,
        )
        hook_kwargs = {
            "event": event,
            "stage": stage,
            "component": component,
            "previous_component": previous_component,
            "next_component": next_component,
            **kwargs,
        }
        plugin_objects = getattr(self, "_composed_hook_plugins", None)
        if not isinstance(plugin_objects, list):
            plugin_objects = self._get_plugins()
        outputs: list[Any] = []
        for hook_name in hook_names:
            for plugin in plugin_objects:
                hook = getattr(plugin, hook_name, None)
                if callable(hook):
                    outputs.append(hook(self, **hook_kwargs))
        return outputs

    def _stage_hook_token(self, stage: str) -> str:
        return self._normalize_score_stage(
            stage,
            allow_all_auto=False,
        ).replace("-", "_")

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
            if isinstance(stage_value, (list, tuple, set)):
                raw_stages.extend([str(token) for token in stage_value])
                continue
            raw_stages.append(str(stage_value))

        if len(raw_stages) == 0:
            return [self.default_stage]
        return raw_stages

    def _expand_canonical_score_stages(self, raw_stages: list[str]) -> list[str]:
        return expand_score_stages(
            raw_stages,
            default_stage=self.default_stage,
            stage_order=self.score_stage_order,
            stage_aliases=self.score_stage_aliases,
        )

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
            stage = self._normalize_score_stage(
                stage,
                allow_all_auto=False,
            )
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
        event = self._normalize_hook_event(when)
        normalized_stage = self._normalize_score_stage(
            stage,
            allow_all_auto=False,
        )
        stage_token = normalized_stage.replace("-", "_")
        stage_kwargs = {"stage": normalized_stage, **kwargs}
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
    "SCORE_STAGE_TO_HOOK",
    "RUNTIME_SPLIT_ALIASES",
    "stage_hook_token",
    "normalize_score_stage",
    "expand_score_stages",
    "normalize_score_mode",
    "normalize_runtime_split_mode",
    "resolve_data_split_payload",
    "resolve_sensitive_split_payload",
    "normalize_keyword_filters",
    "filter_scores_by_keywords",
    "ScoreOrchestratorMixin",
]
