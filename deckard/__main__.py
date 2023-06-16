""""Runs a submodule passed as an arg."""

import argparse
import subprocess
import logging

logger = logging.getLogger(__name__)


def run_submodule(submodule):
    cmd = f"python -m deckard.layers.{submodule}"
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, bufsize=1) as proc:
        for line in proc.stdout:
            logger.info(line)
        # for line in proc.stderr:
        #     logger.error(line)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "submodule", type=str, choices=["experiment", "optimise", "find_best",],
    )
    args = parser.parse_args()
    submodule = args.submodule
    run_submodule(submodule)
