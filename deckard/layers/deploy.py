import logging
import argparse
from pathlib import Path
import yaml
from ..iaac import GCP_Config


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    iaac_parser = argparse.ArgumentParser()
    iaac_parser.add_argument("--verbosity", type=str, default="INFO")
    iaac_parser.add_argument("--config_dir", type=str, default="conf")
    iaac_parser.add_argument("--config_file", type=str, default="default.yaml")
    iaac_parser.add_argument("--workdir", type=str, default=".")
    args = iaac_parser.parse_args()
    config_dir = Path(args.workdir, args.config_dir).resolve().as_posix()
    config_file = Path(config_dir, args.config_file).resolve().as_posix()
    with open(config_file, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    gcp = GCP_Config(**params)
    logging.basicConfig(level=args.verbosity)
    ip_addr = gcp()
    logger.info(f"IP address of the filestore: {ip_addr}")
    command = f"sudo mount -o rw,intr {ip_addr}:/vol1 <mount_directory>"
    logger.info(f"Run command: {command} to mount the filestore.")
