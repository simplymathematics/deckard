import logging
import argparse
import yaml
from os import environ
from dvc.api import params_show
from pathlib import Path

logger = logging.getLogger(__name__)

DECKARD_PIPELINE_FILE = environ.get("DECKARD_PIPELINE_FILE", "dvc.yaml")

# with open(DECKARD_PIPELINE_FILE, "r") as f:
#     pipeline = yaml.safe_load(f)
# stages = list(pipeline["stages"].keys())

if __name__ == "__main__":
    queue_parser = argparse.ArgumentParser(description="Queue example")
    queue_parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    queue_parser.add_argument(
        "--config_file",
        "-c",
        default=["conf/default.yaml"],
        help="Full path to the configuration file",
        nargs="+",
    )
    queue_parser.add_argument(
        "--params_file",
        "-p",
        default=[],
        nargs="+",
    )
    queue_parser.add_argument(
        "--output_file",
        "-o",
        default="queue",
        help="Full path to the output folder",
        required=True,
    )
    # queue_parser.add_argument(
    #     "stage",
    #     help="Stage to run",
    #     # choices=[f"{stage}" for stage in stages],
    # )
    args = queue_parser.parse_args()
    # Set logging level
    logging.basicConfig(level=args.log_level)
    # Load Default Configuration
    def load_config(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config

    # Add overrides from params file
    # stage = pipeline["stages"][args.stage]
    def add_overrides_from_params_file(params_file, default_config):
        params = default_config["hydra"]["sweeper"]["params"]
        if params_file is not None:
            with open(params_file, "r") as f:
                param_file_params = yaml.safe_load(f)
                if (
                    not isinstance(param_file_params, type(None))
                    and len(param_file_params) > 0
                ):
                    params.update(param_file_params)
            logger.info(f"Loaded parameter overrides: {yaml.dump(params)}")
            # Update configuration
            new_config = default_config.copy()
            new_config["hydra"]["sweeper"]["params"].update(params)
            logger.info(
                f"New configuration: {new_config['hydra']['sweeper']['params']}"
            )
        else:
            new_config = default_config
        return new_config

    config_list = [load_config(config_file) for config_file in args.config_file]
    config_names = [Path(config_file).stem for config_file in args.config_file]
    params_list = [Path(params_file) for params_file in args.params_file]
    param_names = [Path(params_file).stem for params_file in args.params_file]
    new_configs = {}
    i = 0
    Path(args.output_file).mkdir(parents=True, exist_ok=True)
    for default_config in config_list:
        conf_name = config_names[i]
        i += 1
        j = 0
        for params_file in params_list:
            param_name = param_names[j]
            j += 1
            new_config = add_overrides_from_params_file(
                params_file=params_file, default_config=default_config
            )
            if conf_name == "default":
                study_name = f"{param_name}"
                new_configs[study_name] = new_config
            else:
                study_name = f"{conf_name}_{param_name}"
                new_configs[study_name] = new_config
            for k, v in new_configs.items():
                v["hydra"]["sweeper"]["params"]["study_name"] = k
                with open(Path(args.output_file, k).with_suffix(".yaml"), "w") as f:
                    yaml.dump(v, f)

    # # Write new configuration
    # Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    # with open(args.output_file, "w") as f:
    #     yaml.dump(new_config, f)
