import logging
from pathlib import Path
import argparse
import os
from omegaconf import OmegaConf
from .utils import save_params_file

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("overrides", type=str, nargs="*", default=None)
parser.add_argument("--verbosity", type=str, default="INFO")
parser.add_argument("--params_file", type=str, default="params.yaml")
parser.add_argument("--config_dir", type=str, default="conf")
parser.add_argument("--config_file", type=str, default="default")
parser.add_argument("--workdir", type=str, default=".")


def main(args) -> None:
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
    args = parser.parse_args()
    main(args)
