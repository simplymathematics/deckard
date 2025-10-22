import warnings
import logging
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate


from .experiment import ExperimentConfig


logger = logging.getLogger(__name__)


# Suppress sklearn runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

supported_modules = ["data", "model", "defend", "attack"]
# Parse the config_dir argument first to set up config dir


def optimize(cfg: ExperimentConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # Pop "optimizers" from cfg if in cfg and use it to return a subset of the scores
    if "optimizers" in cfg and cfg.optimizers is not None:
        optimizers = cfg.optimizers
        cfg.pop("optimizers")
    else:
        optimizers = None
    # Initialize experiment config
    if "_target_" not in cfg:
        cfg["_target_"] = "deckard.experiment.ExperimentConfig"
    experiment = instantiate(cfg)
    scores = experiment()
    if optimizers:
        scores = {k: v for k, v in scores.items() if k in optimizers}
    return scores


def main():
    # Get config dir from environment variable if set
    config_dir = os.environ.get(
        "DECKARD_CONFIG_DIR",
        ValueError("DECKARD_CONFIG_DIR environment variable not set."),
    )
    config_dir = str(Path(config_dir).resolve())
    config_file = os.environ.get("DECKARD_DEFAULT_CONFIG_FILE", "default.yaml")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("Starting Deckard with Hydra configuration.")
    logger.info(f"Config directory: {Path(config_dir).resolve()}")
    config_file = Path(config_dir) / config_file
    logger.info(f"Resolved config file path: {config_file.resolve()}")
    if not config_file.exists():
        logger.error(f"Config file {config_file} does not exist.")
        sys.exit(1)

    @hydra.main(
        config_path=config_dir, config_name=str(config_file.name), version_base="1.3"
    )
    def main_hydra(cfg: ExperimentConfig) -> None:
        scores = optimize(cfg=cfg)

        return scores

    return main_hydra()


if __name__ == "__main__":

    main()
