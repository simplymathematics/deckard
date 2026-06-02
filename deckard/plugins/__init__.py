"""Plugin namespace package.

Framework-agnostic plugin implementations live under plugin-family modules.
Plugin modules are loaded lazily to avoid importing optional dependencies at
package import time.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from .._optional import (
    get_optional_family_module,
    get_optional_family_names,
    get_optional_family_required_imports,
    is_optional_family_available,
)

PluginScalar = str | int | float | bool | None
PluginValue = PluginScalar | list["PluginValue"] | dict[str, "PluginValue"]


class PluginRuntimePlugin(Protocol):
    """Minimal runtime protocol exposing plugin-invoked methods.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def __getattr__(self, name: str) -> Any: ...

    def __call__(
        self,
        *args: PluginValue,
        **kwargs: PluginValue,
    ) -> PluginValue | None:
        """Invoke runtime object as callable plugin target.

        Args:
            *args: Positional runtime payload values.
            **kwargs: Keyword runtime payload values.

        Returns:
            Runtime plugin output payload.
        """
        ...


@dataclass(eq=False, kw_only=True)
class HookPlugin:
    """Generic hook plugin that delegates one runtime hook to one method.

    Args:
            hook_name: Runtime hook name exposed by the plugin.
            method_name: Runtime method name invoked when the hook runs.
            method_kwargs: Default kwargs merged into hook invocation kwargs.
            init_params: Metadata-only declaration payload for docs and tooling.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    hook_name: str
    method_name: str
    method_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Configuration field: method_kwargs."},
    )
    init_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Configuration field: init_params."},
    )

    def declares_hook(self, hook_name: str) -> bool:
        """Return whether this plugin handles the provided hook name.

        Args:
            hook_name: Runtime hook name to compare against this plugin mapping.

        Returns:
            True when the provided hook name matches this plugin hook.
        """
        return hook_name == self.hook_name

    def _invoke(self, runtime: Any, **kwargs: Any):
        method = getattr(runtime, self.method_name, None)
        if not callable(method):
            raise AttributeError(
                f"Runtime '{type(runtime).__name__}' has no callable '{self.method_name}'",
            )
        call_kwargs = dict(self.method_kwargs)
        call_kwargs.update(kwargs)
        return method(**call_kwargs)

    def __call__(
        self,
        runtime: PluginRuntimePlugin,
        *args: PluginValue,
        **kwargs: PluginValue,
    ) -> PluginValue | None:
        """Dispatch runtime hook calls that match this plugin's declared hook.

        Args:
            runtime: Runtime object that owns the configured plugin method.
            *args: Positional hook arguments (currently unused passthrough payload).
            **kwargs: Hook keyword arguments merged into plugin invocation context.

        Returns:
            Hook method output when the hook matches, otherwise None.
        """
        _ = args
        hook_name = kwargs.pop("hook_name", None)
        if hook_name is not None and hook_name != self.hook_name:
            return None
        return self._invoke(runtime, **kwargs)

    def __getattr__(self, attr_name: str):
        try:
            hook_name = object.__getattribute__(self, "hook_name")
        except AttributeError:
            raise AttributeError(attr_name)
        if attr_name != hook_name:
            raise AttributeError(attr_name)

        def _hook(runtime: Any, *args: Any, **kwargs: Any):
            return self(runtime, *args, hook_name=attr_name, **kwargs)

        return _hook


_PLUGIN_NAMES = get_optional_family_names(kind="plugin")


def is_plugin_available(name: str) -> bool:
    """Return whether a plugin family has its optional dependencies installed."""
    if name not in _PLUGIN_NAMES:
        raise KeyError(f"Unknown plugin: {name}")
    return is_optional_family_available(name, kind="plugin")


def get_plugin(name: str):
    """Lazily import a plugin package by family name.

    Raises:
            ImportError: If plugin is unavailable or optional dependency is missing.
            KeyError: If plugin name is unknown.
    """
    if name not in _PLUGIN_NAMES:
        raise KeyError(f"Unknown plugin: {name}")

    if not is_plugin_available(name):
        required = ", ".join(get_optional_family_required_imports(name) or (name,))
        raise ImportError(
            f"Plugin '{name}' is not available. Install optional dependencies "
            f"for it ({required}) or the matching deckard extra.",
        )

    try:
        return get_optional_family_module(name, kind="plugin")
    except ImportError as e:
        raise ImportError(
            f"Plugin '{name}' is not available. "
            f"Install optional dependencies for it (e.g. deckard[{name}]).",
        ) from e


def __getattr__(name: str):
    """Lazily resolve plugin-family module attributes.

    Allows attribute access like ``deckard.plugins.fairlearn`` without eagerly
    importing optional plugin dependencies at package import time.
    """
    if name in _PLUGIN_NAMES:
        return get_plugin(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HookPlugin",
    "get_plugin",
    "is_plugin_available",
    *list(_PLUGIN_NAMES),
]
