"""TextAttack plugin exports."""

from importlib import import_module

_SYMBOL_MODULES = {
    "run_textattack_attack_config": ".attack",
    "TextAttackConfig": ".attack",
}


def __getattr__(name: str):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "run_textattack_attack_config",
    "TextAttackConfig",
]
