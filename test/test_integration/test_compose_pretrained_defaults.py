from .shared_compose import PYTORCH_CONFIG_DIR, SKLEARN_CONFIG_DIR, compose_config


def test_sklearn_pretrained_default_composes_with_attack_and_defense():
    cfg = compose_config(SKLEARN_CONFIG_DIR, "pretrained-default")

    assert cfg is not None
    assert cfg.model.trainer._target_ == "deckard.model.trainers.PretrainedTrainer"
    assert cfg.attack.alias == "hsj"
    assert cfg.defense.name == "art.defences.postprocessor.ClassLabels"


def test_sklearn_pretrained_default_composes_without_attack_and_defense():
    cfg = compose_config(
        SKLEARN_CONFIG_DIR,
        "pretrained-default",
        overrides=["~attack", "~defense"],
    )

    assert cfg is not None
    assert cfg.model.trainer._target_ == "deckard.model.trainers.PretrainedTrainer"
    assert cfg.get("attack") is None
    assert cfg.get("defense") is None


def test_sklearn_pretrained_default_composes_without_pretrained_trainer():
    cfg = compose_config(
        SKLEARN_CONFIG_DIR,
        "pretrained-default",
        overrides=["trainer@model.trainer=sklearn"],
    )

    assert cfg is not None
    assert cfg.model.trainer._target_ != "deckard.model.trainers.PretrainedTrainer"


def test_pytorch_pretrained_default_composes_with_attack_and_defense():
    cfg = compose_config(PYTORCH_CONFIG_DIR, "pretrained-default")

    assert cfg is not None
    assert cfg.model.trainer._target_ == "deckard.model.trainers.PretrainedTrainer"
    assert cfg.attack.alias == "fgm"
    assert cfg.defense.name == "art.defences.postprocessor.ClassLabels"


def test_pytorch_pretrained_default_composes_without_attack_and_defense():
    cfg = compose_config(
        PYTORCH_CONFIG_DIR,
        "pretrained-default",
        overrides=["~attack", "~defense"],
    )

    assert cfg is not None
    assert cfg.model.trainer._target_ == "deckard.model.trainers.PretrainedTrainer"
    assert cfg.get("attack") is None
    assert cfg.get("defense") is None


def test_pytorch_pretrained_default_composes_without_pretrained_trainer():
    cfg = compose_config(
        PYTORCH_CONFIG_DIR,
        "pretrained-default",
        overrides=["trainer@model.trainer=pytorch"],
    )

    assert cfg is not None
    assert cfg.model.trainer._target_ != "deckard.model.trainers.PretrainedTrainer"
