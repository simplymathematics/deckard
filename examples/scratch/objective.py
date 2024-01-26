import hydra
from omegaconf import DictConfig
from optuna.trial import Trial
from pathlib import Path
import logging
import os

from deckard.layers.optimise import optimise

logger = logging.getLogger(__name__)
config_path = os.environ.pop(
    "DECKARD_CONFIG_PATH", str(Path(Path(), "conf").absolute().as_posix()),
)
config_name = os.environ.pop("DECKARD_DEFAULT_CONFIG", "default.yaml")


@hydra.main(config_path=config_path, config_name=config_name, version_base="1.3")
def hydra_optimise(cfg: DictConfig) -> float:
    score = optimise(cfg)
    return score


def configure(cfg: DictConfig, trial: Trial) -> None:
    data_name = trial.params.get("data.generate.name", None)
    preprocessor = trial.params.get("model.art.pipeline.preprocessor", None)
    postprocessor = trial.params.get("model.art.pipeline.postprocessor", None)
    if data_name in ["torch_cifar10", "torch_mnist"]:
        trial.suggest_int("data.sample.random_state", 0, 10)
    if preprocessor is not None:
        if preprocessor.strip() == "art.defences.preprocessor.FeatureSqueezing":
            bit_depth = trial.suggest_loguniform(
                "model.art.pipeline.preprocessor.defences.FeatureSqueezing.bit_depth",
                1,
                64,
            )
            trial.suggest_categorical(
                "model.art.pipeline.preprocessor.defences.FeatureSqueezing.clip_values",
                [[0, 2 ^ bit_depth - 1]],
            )
        elif preprocessor.strip() == "art.defences.preprocessor.GaussianAugmentation":
            _ = trial.suggest_loguniform(
                "model.art.pipeline.preprocessor.defences.GaussianAugmentation.sigma",
                0.1,
                10,
            )
            trial.suggest_categorical(
                "model.art.pipeline.preprocessor.defences.GaussianAugmentation.clip_values",
                [[0, 255]],
            )
            trial.suggest_categorical(
                "model.art.pipeline.preprocessor.defences.GaussianAugmentation.ratio",
                [0.5],
            )
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor}")
    if postprocessor is not None:
        if postprocessor.strip() == "art.defences.postprocessor.HighConfidence":
            _ = trial.suggest_int(
                "model.art.pipeline.preprocessor.defences.HighConfidence.threshold", 1,
            )
            trial.suggest_categorical(
                "model.art.pipeline.preprocessor.defences.HighConfidence.abs", [True],
            )
            trial.suggest_categorical(
                "model.art.pipeline.preprocessor.defences.HighConfidence.clip_values",
                [[0, 255]],
            )
        elif postprocessor.strip() == "art.defences.postprocessor.GaussianNoise":
            _ = trial.suggest_loguniform(
                "model.art.pipeline.preprocessor.defences.GaussianNoise.sigma", 0.1, 10,
            )
            trial.suggest_categorical(
                "model.art.pipeline.preprocessor.defences.GaussianNoise.clip_values",
                [[0, 255]],
            )
        else:
            raise ValueError(f"Unknown preprocessor {postprocessor}")


if __name__ == "__main__":
    hydra_optimise()
