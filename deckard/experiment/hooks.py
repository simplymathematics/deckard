"""Shared helpers for experiment plugin-hook orchestration."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from ..plugins import HookPlugin


def normalize_hook_token(value: Any) -> str:
    """Normalize hook event/position token values."""
    return str(value).strip().lower()


def should_run_persist_tail(
    *,
    position_token: str,
    event_token: str,
    stage_token: str,
) -> bool:
    """Return True when a hook invocation matches the persist tail phase."""
    return (
        position_token == "last"
        and event_token == "after"
        and stage_token == "persist"
    )


def build_plugin_hook_wrappers(
    *,
    stages: Iterable[Any],
    stage_token_resolver: Callable[[Any], str],
    method_name: str,
    plugin_payload_key: str,
    plugin_payload: dict[str, Any],
) -> tuple[list[HookPlugin], list[HookPlugin]]:
    """Build paired first/last plugin hook wrappers for each stage/event."""
    first_hooks: list[HookPlugin] = []
    last_hooks: list[HookPlugin] = []
    for stage in stages:
        stage_token = stage_token_resolver(stage)
        for event in ("before", "after"):
            hook_name = f"{event}_{stage_token}"
            first_hooks.append(
                HookPlugin(
                    hook_name=hook_name,
                    method_name=method_name,
                    method_kwargs={
                        plugin_payload_key: plugin_payload,
                        "plugin_position": "first",
                    },
                ),
            )
            last_hooks.append(
                HookPlugin(
                    hook_name=hook_name,
                    method_name=method_name,
                    method_kwargs={
                        plugin_payload_key: plugin_payload,
                        "plugin_position": "last",
                    },
                ),
            )
    return first_hooks, last_hooks


def build_hook_run_result(
    *,
    enabled: bool,
    position_token: str,
    component: Any,
    stage_token: str,
    event_token: str,
) -> dict[str, Any]:
    """Build canonical hook execution metadata payload."""
    return {
        "enabled": bool(enabled),
        "position": position_token,
        "component": str(component),
        "stage": stage_token,
        "event": event_token,
        "executed": False,
    }


def prepare_hook_run_context(
    *,
    enabled: bool,
    component: Any,
    stage: Any,
    event: Any,
    plugin_position: Any,
    resolve_stage_token: Callable[[Any], str],
) -> tuple[str, str, str, dict[str, Any]]:
    """Return normalized stage/event/position tokens and base result payload."""
    stage_token = resolve_stage_token(stage)
    event_token = normalize_hook_token(event)
    position_token = normalize_hook_token(plugin_position)
    result = build_hook_run_result(
        enabled=enabled,
        position_token=position_token,
        component=component,
        stage_token=stage_token,
        event_token=event_token,
    )
    return stage_token, event_token, position_token, result
