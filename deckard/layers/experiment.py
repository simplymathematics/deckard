#! /usr/bin/env python

import logging
from pathlib import Path
import argparse


from .utils import run_stages

logger = logging.getLogger(__name__)

experiment_parser = argparse.ArgumentParser()
experiment_parser.add_argument("stage", type=str, nargs="*", default=None)
experiment_parser.add_argument("--verbosity", type=str, default="INFO")
experiment_parser.add_argument("--params_file", type=str, default="params.yaml")
experiment_parser.add_argument("--pipeline_file", type=str, default="dvc.yaml")
experiment_parser.add_argument("--config_dir", type=str, default="conf")
experiment_parser.add_argument("--config_file", type=str, default="default")
experiment_parser.add_argument("--workdir", type=str, default=".")
experiment_parser.add_argument("--overrides", nargs="*", default=[], type=str)


def experiment_main(args):
    config_dir = Path(args.workdir, args.config_dir).absolute().as_posix()
    logging.basicConfig(
        level=args.verbosity,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if args.overrides is not None and len(args.overrides) > 0:
        for override in args.overrides:
            key, value = override.split("=")
            logger.info(f"Setting {key}={value}")
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
        overrides=args.overrides,
    )
    return results


if __name__ == "__main__":
    args = experiment_parser.parse_args()
    experiment_main(args)
