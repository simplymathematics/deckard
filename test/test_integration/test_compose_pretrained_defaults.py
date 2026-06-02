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


def _reset_hydra_state() -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_store = ConfigStore.instance()
    for key in list(config_store.repo.keys()):
        if key not in {"hydra", "_dummy_empty_config_.yaml"}:
            config_store.repo.pop(key, None)


def _compose(config_dir: Path, config_name: str, overrides: list[str] | None = None):
    _reset_hydra_state()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=overrides or [])


def test_sklearn_pretrained_default_composes_with_attack_and_defense():
    cfg = _compose(SKLEARN_CONFIG_DIR, "pretrained-default")

    assert cfg is not None
    assert cfg.model.trainer._target_ == "deckard.model.trainers.PretrainedTrainer"
    assert cfg.attack.alias == "hsj"
    assert cfg.defense.name == "art.defences.postprocessor.ClassLabels"


def test_sklearn_pretrained_default_composes_without_attack_and_defense():
    cfg = _compose(
        SKLEARN_CONFIG_DIR,
        "pretrained-default",
        overrides=["~attack", "~defense"],
    )

    assert cfg is not None
    assert cfg.model.trainer._target_ == "deckard.model.trainers.PretrainedTrainer"
    assert cfg.get("attack") is None
    assert cfg.get("defense") is None


def test_sklearn_pretrained_default_composes_without_pretrained_trainer():
    cfg = _compose(
        SKLEARN_CONFIG_DIR,
        "pretrained-default",
        overrides=["trainer@model.trainer=sklearn"],
    )

    assert cfg is not None
    assert cfg.model.trainer._target_ != "deckard.model.trainers.PretrainedTrainer"


def test_pytorch_pretrained_default_composes_with_attack_and_defense():
    cfg = _compose(PYTORCH_CONFIG_DIR, "pretrained-default")

    assert cfg is not None
    assert cfg.model.trainer._target_ == "deckard.model.trainers.PretrainedTrainer"
    assert cfg.attack.alias == "fgm"
    assert cfg.defense.name == "art.defences.postprocessor.ClassLabels"


def test_pytorch_pretrained_default_composes_without_attack_and_defense():
    cfg = _compose(
        PYTORCH_CONFIG_DIR,
        "pretrained-default",
        overrides=["~attack", "~defense"],
    )

    assert cfg is not None
    assert cfg.model.trainer._target_ == "deckard.model.trainers.PretrainedTrainer"
    assert cfg.get("attack") is None
    assert cfg.get("defense") is None


def test_pytorch_pretrained_default_composes_without_pretrained_trainer():
    cfg = _compose(
        PYTORCH_CONFIG_DIR,
        "pretrained-default",
        overrides=["trainer@model.trainer=pytorch"],
    )

    assert cfg is not None
    assert cfg.model.trainer._target_ != "deckard.model.trainers.PretrainedTrainer"
