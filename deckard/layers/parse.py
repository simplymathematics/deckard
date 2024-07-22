import logging
from pathlib import Path
import argparse
import os
from omegaconf import OmegaConf
from .utils import save_params_file

__all__ = ["parse_hydra_config", "hydra_parser"]

logger = logging.getLogger(__name__)
hydra_parser = argparse.ArgumentParser()
hydra_parser.add_argument("overrides", type=str, nargs="*", default=None)
hydra_parser.add_argument("--verbosity", type=str, default="INFO")
hydra_parser.add_argument("--params_file", type=str, default="params.yaml")
hydra_parser.add_argument("--config_dir", type=str, default="conf")
hydra_parser.add_argument("--config_file", type=str, default="default")
hydra_parser.add_argument("--workdir", type=str, default=".")


def parse_hydra_config(args) -> None:
    logging.basicConfig(level=args.verbosity)
    config_dir = Path(Path(), args.config_dir).resolve().as_posix()
    OmegaConf.register_new_resolver("eval", eval)
    assert (
        save_params_file(
            config_dir=config_dir,
            config_file=args.config_file,
            params_file=args.params_file,
            overrides=args.overrides,
        )
        is None
    )
    os.environ["DECKARD_DEFAULT_CONFIG"] = args.config_file
    os.environ["DECKARD_CONFIG_PATH"] = args.config_dir
    return None


if __name__ == "__main__":
    args = hydra_parser.parse_args()
    parse_hydra_config(args)
