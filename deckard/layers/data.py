#! /usr/bin/env python

import logging
from pathlib import Path
import argparse


from .utils import save_params_file, run_stages

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("stage", type=str, nargs="*", default=None)
parser.add_argument("--verbosity", type=str, default="INFO")
parser.add_argument("--params_file", type=str, default="params.yaml")
parser.add_argument("--pipeline_file", type=str, default="dvc.yaml")
parser.add_argument("--config_dir", type=str, default="conf")
parser.add_argument("--config_file", type=str, default="default")
parser.add_argument("--workdir", type=str, default=".")
parser.add_argument("--overrides", nargs="*", default=[], type=str)


def main(args):
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
        sub_dict="data",
    )
    return results


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
