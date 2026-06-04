"""Shared plugin runtime and score orchestration mixins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import HookPlugin
from ..artifacts import ScoreDict
from ..utils import (
    coerce_to_list,
    instantiate_plugin_spec,
    load_class,
    normalize_plugin_specs,
)


def _clone_hook_plugin(plugin: HookPlugin) -> HookPlugin:
    """Return a detached hook plugin instance for safe per-runtime composition."""
    return HookPlugin(
        hook_name=plugin.hook_name,
        method_name=plugin.method_name,
        method_kwargs=dict(plugin.method_kwargs),
        init_params=dict(plugin.init_params),
    )


@dataclass(frozen=True)
class HookBundle:
    """Named reusable collection of hook plugins for runtime composition."""

    name: str
    hooks: tuple[HookPlugin, ...]

    def clone_plugins(self) -> list[HookPlugin]:
        """Return detached plugin copies for safe per-runtime composition.

        Returns:
                Cloned hook plugins.
        """
        return [_clone_hook_plugin(plugin) for plugin in self.hooks]


def compose_hook_plugins(*parts: Any) -> list[HookPlugin]:
    """Compose hook bundles and plugins into a deduplicated runtime list.

    The function accepts ``HookBundle`` instances, individual ``HookPlugin``
    objects, or nested lists/tuples containing either form. Nested bundles are
    flattened, plugin instances are cloned before return so call sites do not
    share mutable plugin state, and duplicates are removed by the
    ``(hook_name, method_name)`` pair while preserving first-seen order.

    Args:
        *parts: Hook bundles, plugins, or nested iterables of those values.

    Returns:
        Ordered hook plugin list with duplicates removed.

    Raises:
        TypeError: If any nested value is not a ``HookBundle`` or
            ``HookPlugin``.
    """
    plugins: list[HookPlugin] = []
    seen: set[tuple[str, str]] = set()

    def _append(plugin: HookPlugin) -> None:
        key = (plugin.hook_name, plugin.method_name)
        if key in seen:
            return
        seen.add(key)
        plugins.append(plugin)

    for part in parts:
        if part is None:
            continue
        if isinstance(part, HookBundle):
            for plugin in part.clone_plugins():
                _append(plugin)
            continue
        if isinstance(part, HookPlugin):
            _append(_clone_hook_plugin(part))
            continue
        for item in coerce_to_list(part):
            if isinstance(item, HookBundle):
                for plugin in item.clone_plugins():
                    _append(plugin)
            elif isinstance(item, HookPlugin):
                _append(_clone_hook_plugin(item))
            else:
                raise TypeError(
                    "compose_hook_plugins accepts HookPlugin, HookBundle, or lists of them",
                )
    return plugins


@dataclass(eq=False, kw_only=True)
class RuntimeBase:
    """Reusable plugin instantiation and hook dispatch behavior."""

    def _instantiate_plugin(self, plugin_spec: Any):
        """Instantiate one runtime plugin specification.

        Args:
            plugin_spec: Plugin declaration payload or runtime plugin object.

        Returns:
            Instantiated plugin object.
        """
        return instantiate_plugin_spec(plugin_spec, loader=load_class)

    def _get_plugins(self) -> list:
        """Resolve and cache runtime plugins for this object instance.

        Returns:
            Ordered list of instantiated runtime plugins.
        """
        if not hasattr(self, "_plugin_objects") or self._plugin_objects is None:
            plugin_specs = normalize_plugin_specs(getattr(self, "plugins", []))
            self._plugin_objects = [
                self._instantiate_plugin(spec) for spec in plugin_specs
            ]
        return self._plugin_objects

    def _run_plugin_hook(self, hook_name: str, **kwargs: Any) -> list[Any]:
        """Execute one plugin hook across all instantiated runtime plugins.

        Args:
            hook_name: Hook method name to invoke when present on a plugin.
            **kwargs: Hook-specific keyword arguments.

        Returns:
            Ordered list of hook return values.
        """
        hook_outputs: list[Any] = []
        for plugin in self._get_plugins():
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                hook_outputs.append(hook(self, **kwargs))
        return hook_outputs

    def _merge_plugin_scores(self, hook_outputs: list[Any]) -> None:
        """Merge dictionary-like hook outputs into the runtime score payload."""
        current_scores = ScoreDict.from_payload(getattr(self, "score_dict", {}) or {})
        for output in hook_outputs:
            if isinstance(output, dict):
                current_scores.update(ScoreDict.from_payload(output))
        self.score_dict = current_scores


__all__ = [
    "HookBundle",
    "compose_hook_plugins",
    "RuntimeBase",
]
