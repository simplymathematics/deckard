from pathlib import Path
import logging
import os
import yaml
from functools import reduce
from operator import mul
from ..base.utils import make_grid, my_hash

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

def find_configs_in_folder(conf_dir, regex):
    configs = []
    for path in Path(conf_dir).rglob(regex):
        configs.append(path)
    logger.info(f"Found {len(configs)} configs in {conf_dir}")
    return configs

def find_config_folders(conf_dir):
    config_folders = []
    for path in Path(conf_dir).rglob("*"):
        if path.is_dir():
            config_folders.append(path)
    logger.info(f"Found {len(config_folders)} config folders in {conf_dir}")
    return config_folders


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
    logger.debug(f"Loaded config from {config_path}")
    return config


def dict_to_overrides(dictionary):
    new = {}
    for key, value in dictionary.items():
        for k, v in value.items():
            new_key = "++" + key + "." + k
            new[new_key] = v
    return new


def generate_grid_from_folders(conf_dir, regex):
    this_dir = os.getcwd()
    conf_dir = os.path.relpath(conf_dir, this_dir)
    config_folders = find_config_folders(conf_dir)
    big_dict = {}
    layers = []
    for folder in config_folders:
        folder_key = os.path.relpath(folder, conf_dir)
        big_dict[folder_key] = [] if folder not in big_dict else big_dict[folder_key]
        config_paths = find_configs_in_folder(folder, regex)
        for config_path in config_paths:
            config = load_config(config_path)
            if isinstance(config, type(None)) or len(config) == 0:
                big_dict[folder_key].append({})
                continue
            big_dict[folder_key].append(config)
        layers.append(len(big_dict[folder_key]))
    big_list = make_grid(big_dict)

    assert len(big_list) == reduce(
        mul,
        layers,
    ), f"Grid size {len(big_list)} does not match product of layer sizes {reduce(mul, layers)}"
    logger.info(f"Generated grid with {len(big_list)} configs")
    return big_list


def generate_queue(
    conf_root,
    grid_dir,
    regex,
    queue_folder="queue",
    default_file="default.yaml",
):
    this_dir = os.getcwd()
    conf_dir = os.path.join(this_dir, conf_root, grid_dir)
    logger.debug(f"Looking for configs in {conf_dir}")
    big_list = generate_grid_from_folders(conf_dir, regex)
    i = 0
    for entry in big_list:
        new = dict_to_overrides(entry)
        path = Path(conf_root, queue_folder)
        name = my_hash(entry)
        path.mkdir(parents=True, exist_ok=True)
        with open(Path(conf_root, default_file), "r") as stream:
            try:
                default = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(exc)
        default["hydra"]["sweeper"]["params"] = new
        big_list[i] = default
        with open(Path(path, name + ".yaml"), "w") as outfile:
            yaml.dump(big_list[i], outfile, default_flow_style=False)
        assert Path(path, name + ".yaml").exists()
        i += 1
    return big_list


conf_root = "conf"
grid_folder = "grid"
regex = "*.yaml"

big_list = generate_queue(conf_root, grid_folder, regex)
print(yaml.dump(big_list[0]))
