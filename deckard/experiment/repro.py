"""DVC persistence plugin helpers for experiment runtime contracts.

This module isolates reproducibility/persistence concerns (params snapshots,
DVC pull/push, and pipeline autogeneration) from DVCLive monitoring/logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..plugins import HookPlugin
from . import dvc as _dvc_module
from .canon import CANONICAL_EXPERIMENT_PIPELINE_STAGES
from .dvc import (
    _resolve_stage_token,
    build_dvc_cmd as _build_dvc_cmd,
    build_dvc_stage_plan as _build_dvc_stage_plan,
    coerce_dvc_experiment_plugin,
    extract_dvc_file_aliases as _extract_dvc_file_aliases,
    generate_dvc_pipeline as _generate_dvc_pipeline,
)

PluginPrimitive = str | int | float | bool | None


@dataclass(eq=False, kw_only=True)
class DVCReproPlugin:
    """Runtime DVC persistence policy for experiment orchestration hooks.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    enabled: bool = False
    mode: str = "single"
    pull_dependencies: bool = True
    push_outputs: bool = True
    fail_on_dvc_error: bool = False
    dvc_file: str = "dvc.yaml"
    params_file: str = "params.yaml"

    def __call__(
        self,
        *args: Any,
        **overrides: Any,
    ) -> dict[str, PluginPrimitive]:
        """Return normalized plugin payload, optionally applying runtime overrides.

        Args:
            *args: Unused positional plugin hook arguments.
            **overrides: Runtime payload overrides.

        Returns:
            Normalized plugin payload dictionary.
        """
        _ = args
        payload = self.to_dict()
        for key, value in overrides.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[str(key)] = value
            else:
                payload[str(key)] = str(value)
        return payload

    def to_dict(self) -> dict[str, PluginPrimitive]:
        """Serialize repro plugin policy to a runtime payload mapping.

        Returns:
            Plugin payload dictionary.
        """
        return {
            "enabled": bool(self.enabled),
            "mode": str(self.mode),
            "pull_dependencies": bool(self.pull_dependencies),
            "push_outputs": bool(self.push_outputs),
            "fail_on_dvc_error": bool(self.fail_on_dvc_error),
            "dvc_file": str(self.dvc_file),
            "params_file": str(self.params_file),
        }


def coerce_dvc_repro_plugin(plugin: Any) -> DVCReproPlugin:
    """Normalize repro plugin declarations from bool/dict/object forms."""
    if isinstance(plugin, DVCReproPlugin):
        return plugin
    if plugin in [None, False]:
        return DVCReproPlugin(enabled=False)
    if plugin is True:
        return DVCReproPlugin(enabled=True)
    if isinstance(plugin, Mapping):
        payload = dict(plugin)
        return DVCReproPlugin(
            enabled=bool(payload.get("enabled", True)),
            mode=str(payload.get("mode", "single")),
            pull_dependencies=bool(payload.get("pull_dependencies", True)),
            push_outputs=bool(payload.get("push_outputs", True)),
            fail_on_dvc_error=bool(payload.get("fail_on_dvc_error", False)),
            dvc_file=str(payload.get("dvc_file", "dvc.yaml")),
            params_file=str(payload.get("params_file", "params.yaml")),
        )
    raise TypeError(
        "repro_plugin must be a bool, mapping, DVCReproPlugin, or None.",
    )


def _build_repro_plugin_hook_wrappers(
    plugin_cfg: DVCReproPlugin,
    *,
    method_name: str,
) -> tuple[list[HookPlugin], list[HookPlugin]]:
    first_hooks: list[HookPlugin] = []
    last_hooks: list[HookPlugin] = []
    plugin_payload = plugin_cfg.to_dict()

    for stage in CANONICAL_EXPERIMENT_PIPELINE_STAGES:
        stage_token = str(stage).strip().lower().replace("-", "_")
        for event in ("before", "after"):
            hook_name = f"{event}_{stage_token}"
            first_hooks.append(
                HookPlugin(
                    hook_name=hook_name,
                    method_name=method_name,
                    method_kwargs={
                        "repro_plugin": plugin_payload,
                        "plugin_position": "first",
                    },
                ),
            )
            last_hooks.append(
                HookPlugin(
                    hook_name=hook_name,
                    method_name=method_name,
                    method_kwargs={
                        "repro_plugin": plugin_payload,
                        "plugin_position": "last",
                    },
                ),
            )

    return first_hooks, last_hooks


def build_repro_experiment_plugin_hooks(
    plugin: Any,
    *,
    method_name: str = "_repro_experiment_plugin_hook",
) -> tuple[list[HookPlugin], list[HookPlugin]]:
    """Construct first/last DVC persistence hook wrappers."""
    plugin_cfg = coerce_dvc_repro_plugin(plugin)
    if not plugin_cfg.enabled:
        return [], []
    return _build_repro_plugin_hook_wrappers(plugin_cfg, method_name=method_name)


def run_repro_experiment_plugin_hook(
    experiment: Any,
    *,
    repro_plugin: Any,
    plugin_position: str,
    component: str,
    stage: str,
    event: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one DVC persistence hook callback."""
    _ = kwargs

    plugin_cfg = coerce_dvc_repro_plugin(repro_plugin)
    stage_token = _resolve_stage_token(stage)
    event_token = str(event).strip().lower()
    position_token = str(plugin_position).strip().lower()

    result: dict[str, Any] = {
        "enabled": bool(plugin_cfg.enabled),
        "position": position_token,
        "component": str(component),
        "stage": stage_token,
        "event": event_token,
        "executed": False,
    }

    if not plugin_cfg.enabled:
        return result

    # Bridge repro policy into the existing DVC policy shape expected by shared
    # params/pull/push helpers.
    dvc_policy = coerce_dvc_experiment_plugin(plugin_cfg.to_dict())

    if position_token == "first" and event_token == "before" and stage_token == "load":
        result["params_file"] = _dvc_module._write_dvc_params_file(
            experiment,
            plugin=dvc_policy,
            stage=stage_token,
        )
        result["pull"] = _dvc_module._run_dvc_pull(experiment, dvc_policy)
        result["executed"] = True

    if (
        position_token == "last"
        and event_token == "after"
        and stage_token == "persist"
    ):
        result["params_file"] = _dvc_module._write_dvc_params_file(
            experiment,
            plugin=dvc_policy,
            stage=stage_token,
        )
        result["push"] = _dvc_module._run_dvc_push(experiment, dvc_policy)
        result["executed"] = True

    return result


# Backward-compatible persistence API now hosted here.
def extract_dvc_file_aliases(*args: Any, **kwargs: Any) -> dict[str, str]:
    """Proxy to DVC file-alias extraction helper."""
    return _extract_dvc_file_aliases(*args, **kwargs)


def build_dvc_cmd(*args: Any, **kwargs: Any) -> str:
    """Proxy to DVC command construction helper."""
    return _build_dvc_cmd(*args, **kwargs)


def build_dvc_stage_plan(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Proxy to DVC stage-plan construction helper."""
    return _build_dvc_stage_plan(*args, **kwargs)


def generate_dvc_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Proxy to DVC pipeline generation helper."""
    return _generate_dvc_pipeline(*args, **kwargs)


__all__ = [
    "DVCReproPlugin",
    "coerce_dvc_repro_plugin",
    "build_repro_experiment_plugin_hooks",
    "run_repro_experiment_plugin_hook",
    "extract_dvc_file_aliases",
    "build_dvc_cmd",
    "build_dvc_stage_plan",
    "generate_dvc_pipeline",
]
