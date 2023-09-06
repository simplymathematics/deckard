import logging
from pathlib import Path
import argparse
from .utils import save_params_file

logger = logging.getLogger(__name__)
hydra_parser = argparse.ArgumentParser()
hydra_parser.add_argument("overrides", type=str, nargs="*", default=None)
hydra_parser.add_argument("--verbosity", type=str, default="INFO")
hydra_parser.add_argument("--params_file", type=str, default="params.yaml")
hydra_parser.add_argument("--config_dir", type=str, default="conf")
hydra_parser.add_argument("--config_file", type=str, default="default")
hydra_parser.add_argument("--workdir", type=str, default=".")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    
    args = hydra_parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    config_dir = Path(Path(), args.config_dir).resolve().as_posix()
    assert (
        save_params_file(
            config_dir=config_dir,
            config_file=args.config_file,
            params_file=args.params_file,
            overrides=args.overrides,
        )
        is None
    )
