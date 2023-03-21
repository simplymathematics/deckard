import argparse
import logging
import subprocess
import os
logger = logging.getLogger(__name__)

if "__name__" == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", type=str, default="INFO", help="Verbosity of logging. Options are 'DEBUG', 'INFO', 'WARNING', 'ERROR', and 'CRITICAL'.")
    args = parser.parse_args()
    command = f"python -m deckard.layers --layer parse"
    logger.info(f"Running {command}")
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    command = f"dvc repro -f {os.getcwd()}/dvc.yaml"
