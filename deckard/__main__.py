import argparse, logging, subprocess

logger = logging.getLogger(__name__)

if "__name__" == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_add_argument("layer", help="Layer to run.")

    command = f"python -m {parser.parse_args().layer}"
    logger.info(f"Running {command}")
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
