"""Datasets plugin package exports."""

from importlib import import_module

from .. import is_plugin_available

_SYMBOL_MODULES = {
    "FlexibleHuggingFaceDataset": ".declarations",
}


def __getattr__(name: str):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if not is_plugin_available("datasets"):
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} because optional "
            "dependencies for 'datasets' are not installed",
        )
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "FlexibleHuggingFaceDataset",
]
