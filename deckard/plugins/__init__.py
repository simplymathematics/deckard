"""Plugin namespace package.

Framework-agnostic plugin implementations are migrated here by plugin family.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False, kw_only=True)
class HookPlugin:
	"""Generic hook plugin that delegates one runtime hook to one method.

	Args:
		hook_name: Runtime hook name exposed by the plugin.
		method_name: Runtime method name invoked when the hook runs.
		method_kwargs: Default kwargs merged into hook invocation kwargs.
		init_params: Metadata-only declaration payload for docs and tooling.
	"""

	hook_name: str
	method_name: str
	method_kwargs: dict[str, Any] = field(default_factory=dict)
	init_params: dict[str, Any] = field(default_factory=dict)

	def declares_hook(self, hook_name: str) -> bool:
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

	def __call__(self, runtime: Any, *args: Any, **kwargs: Any):
		_ = args
		hook_name = kwargs.pop("hook_name", None)
		if hook_name is not None and hook_name != self.hook_name:
			return None
		return self._invoke(runtime, **kwargs)

	def __getattr__(self, attr_name: str):
		if attr_name != self.hook_name:
			raise AttributeError(attr_name)

		def _hook(runtime: Any, *args: Any, **kwargs: Any):
			return self(runtime, *args, hook_name=attr_name, **kwargs)

		return _hook


__all__ = ["HookPlugin", "anjana", "fairlearn", "lifelines", "seaborn", "yellowbrick"]
"""
Optional plugin system.

Do NOT import plugin modules here at package import time.
Plugins are imported lazily to avoid hard dependencies.
"""

from importlib import import_module

__all__ = [
    "get_plugin",
]


_PLUGIN_MAP = {
    "anjana": "deckard.plugins.anjana",
    "fairlearn": "deckard.plugins.fairlearn",
    "lifelines": "deckard.plugins.lifelines",
    "seaborn": "deckard.plugins.seaborn",
    "yellowbrick": "deckard.plugins.yellowbrick",
}


def get_plugin(name: str):
    """
    Lazily import a plugin package.

    Raises:
        ImportError: if plugin is not installed or missing dependency.
        KeyError: if plugin name is unknown.
    """
    if name not in _PLUGIN_MAP:
        raise KeyError(f"Unknown plugin: {name}")

    module_name = _PLUGIN_MAP[name]

    try:
        return import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Plugin '{name}' is not available. "
            f"Install optional dependencies for it (e.g. deckard[{name}])."
        ) from e