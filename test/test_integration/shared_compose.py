from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra

SKLEARN_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
)
PYTORCH_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "pytorch" / "config"
)


def reset_hydra_compose_state() -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_store = ConfigStore.instance()
    for key in list(config_store.repo.keys()):
        if key not in {"hydra", "_dummy_empty_config_.yaml"}:
            config_store.repo.pop(key, None)


def compose_config(
    config_dir: Path,
    config_name: str,
    overrides: list[str] | None = None,
):
    reset_hydra_compose_state()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=overrides or [])


def compose_sklearn(config_name: str, overrides: list[str] | None = None):
    return compose_config(SKLEARN_CONFIG_DIR, config_name, overrides=overrides)


def compose_pytorch(config_name: str, overrides: list[str] | None = None):
    return compose_config(PYTORCH_CONFIG_DIR, config_name, overrides=overrides)
