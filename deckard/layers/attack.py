#! /usr/bin/env python

import logging
from pathlib import Path
import argparse


from .utils import save_params_file, run_stages

logger = logging.getLogger(__name__)

attack_parser = argparse.ArgumentParser()
attack_parser.add_argument("stage", type=str, nargs="*", default=None)
attack_parser.add_argument("--verbosity", type=str, default="INFO")
attack_parser.add_argument("--params_file", type=str, default="params.yaml")
attack_parser.add_argument("--pipeline_file", type=str, default="dvc.yaml")
attack_parser.add_argument("--config_dir", type=str, default="conf")
attack_parser.add_argument("--config_file", type=str, default="default")
attack_parser.add_argument("--workdir", type=str, default=".")
attack_parser.add_argument("--overrides", nargs="*", default=[], type=str)


def attack_main(args):
    config_dir = Path(args.workdir, args.config_dir).absolute().as_posix()
    logging.basicConfig(
        level=args.verbosity,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if args.overrides is not None and len(args.overrides) > 0:
        save_params_file(
            config_dir=config_dir,
            config_file=args.config_file,
            params_file=args.params_file,
            overrides=args.overrides,
        )
    logger.info(
        f"Using existing params file {args.params_file} in directory {args.workdir}",
    )
    results = run_stages(
        stages=args.stage,
        pipeline_file=args.pipeline_file,
        params_file=args.params_file,
        repo=args.workdir,
        config_dir=config_dir,
        config_file=args.config_file,
        sub_dict="attack",
    )
    return results


if __name__ == "__main__":
    args = attack_parser.parse_args()
    attack_main(args)
