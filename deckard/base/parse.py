import importlib
import logging
import os
from copy import deepcopy
from pathlib import Path
from random import choice
from typing import Union

import yaml
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from .data import Data
from .hashable import my_hash
from .scorer import Scorer

# specify the logger
logger = logging.getLogger(__name__)

# __all__ = [
#     "generate_tuple_list_from_yml",
#     "generate_object_list_from_tuple",
#     "generate_tuple_from_yml",
#     "generate_object_from_tuple",
#     "make_output_folder",
# ]


def generate_tuple_list_from_yml(filename: str) -> list:
    """
    Parses a yml file, generates a an exhaustive list of parameter combinations for each entry in the list, and returns a single list of tuples.
    """
    assert isinstance(
        filename,
        (str, Path, dict),
    ), "filename must be a string, Path, or dict. It is a {}".format(type(filename))
    assert os.path.isfile(filename), f"{filename} does not exist"
    full_list = []
    LOADER = yaml.FullLoader
    # check if the file exists
    if not os.path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    with open(filename, "r") as stream:
        try:
            yml_list = yaml.load(stream, Loader=LOADER)
        except yaml.YAMLError as exc:
            logger.warning("Exception: {}".format(exc))
            raise ValueError("Error parsing yml file {}".format(filename))
    for entry in yml_list:
        if not isinstance(entry, dict):
            raise ValueError("Error parsing yml file {}".format(filename))
        # Popping single parameters, tuples, etc, before the grid search
        special_keys = {}
        for key, value in entry["params"].items():
            if isinstance(value, (tuple, float, int, str)):
                special_values = value
                special_key = key
                special_keys[special_key] = special_values
        for key in special_keys.keys():
            entry["params"].pop(key)
        # Generate the grid search
        grid = ParameterGrid(entry["params"])
        name = entry["name"]
        for combination in grid:
            if "special_keys" in locals():
                for key, value in special_keys.items():
                    combination[key] = value
            full_list.append((name, combination))
    return full_list


def generate_object_list_from_tuple(yml_tuples: list, *args) -> list:
    """
    Imports and initializes objects from yml file. Returns a list of instantiated objects.
    :param yml_list: list of yml entries
    """
    obj_list = []
    for entry in yml_tuples:
        obj_ = generate_object_from_tuple(entry, *args)
        obj_list.append(obj_)
    assert len(obj_list) == len(yml_tuples), "Error instantiating objects"
    return obj_list


def generate_tuple_from_yml(filename: Union[str, dict]) -> list:
    """
    Parses a yml file, generates a an exhaustive list of parameter combinations for each entry in the list, and returns a single list of tuples.
    """
    assert isinstance(
        filename,
        (str, Path, dict),
    ), "filename must be a string, Path, or dict. It is a {}".format(type(filename))
    if isinstance(filename, str):
        LOADER = yaml.FullLoader
        assert os.path.isfile(filename), f"{filename} does not exist"
        with open(filename, "r") as stream:
            try:
                entry = yaml.load(stream, Loader=LOADER)
            except yaml.YAMLError as exc:
                logger.warning("Exception: {}".format(exc))
                raise ValueError("Error parsing yml file {}".format(filename))
    if not os.path.isfile(str(filename)):
        assert isinstance(filename, dict), "filename must be a dict or a yml file"
        entry = filename
    return (entry["name"], entry["params"])


def generate_object_from_tuple(obj_tuple: list, *args) -> list:
    """
    Imports and initializes objects from yml file. Returns a list of instantiated objects.
    :param yml_list: list of yml entries
    """
    library_name = ".".join(obj_tuple[0].split(".")[:-1])
    class_name = obj_tuple[0].split(".")[-1]
    global dependency
    dependency = None
    dependency = importlib.import_module(library_name)
    global deckard_object
    deckard_object = None
    global params
    params = obj_tuple[1]
    exec("from {} import {}".format(library_name, class_name), globals())
    if len(args) == 1:
        global positional_arg
        positional_arg = args[0]
        exec(f"deckard_object = {class_name}(positional_arg, **{params})", globals())
        del positional_arg
    elif len(args) == 0:
        exec(f"deckard_object = {class_name}(**params)", globals())
    else:
        raise ValueError("Too many positional arguments")
    del params
    del dependency
    return deckard_object


def parse_scorer_from_yml(filename: str) -> dict:
    assert isinstance(filename, str)
    LOADER = yaml.FullLoader
    # check if the file exists
    params = {}
    if not os.path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    with open(filename, "r") as stream:
        try:
            scorer_file = yaml.load(stream, Loader=LOADER)[0]
            logger.info(scorer_file)
            logger.info(type(scorer_file))
        except yaml.YAMLError as exc:
            logger.error("Exception: {}".format(exc))
            raise ValueError("Error parsing yml file {}".format(filename))
    # check that datas is a list
    if not isinstance(scorer_file, dict):
        raise ValueError(
            "Error parsing yml file {}. It must be a yaml dictionary.".format(filename),
        )
    if "scorer_function" in scorer_file:
        params["scorer_function"] = scorer_file["scorer_function"]
    elif "name" in scorer_file:
        params["name"] = scorer_file["name"]
    else:
        raise ValueError(
            "Error parsing yml file {}. It must contain a scorer_function or a name.".format(
                filename,
            ),
        )
    logger.info(f"Parsing data from {filename}")
    for param, value in params.items():
        logger.info(param + ": " + str(value))
    data = Scorer(**params)
    assert isinstance(data, Data)
    logger.info("{} successfully parsed.".format(filename))
    return data


def make_output_folder(output_folder: Union[str, Path]) -> Path:
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    global ART_DATA_PATH
    ART_DATA_PATH = output_folder
    assert Path(output_folder).exists(), "Problem creating output folder: {}".format(
        output_folder,
    )
    return Path(output_folder).resolve()


def reproduce_directory_tree(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    input_file: Union[str, Path],
) -> None:
    old_files = [path for path in Path(input_folder).rglob("*" + input_file)]
    old_folders = [path.parent for path in old_files]
    new_folders = [Path(output_folder, path).resolve() for path in old_folders]
    for folder in tqdm(new_folders, desc="Creating Directories"):
        Path(folder).mkdir(parents=True, exist_ok=True)
        assert os.path.isdir(folder.resolve()), "Problem creating folder: {}".format(
            folder,
        )
    return old_files, new_folders


def parse_config(config: Union[dict, str, Path], **kwargs) -> object:
    tuple_ = generate_tuple_from_yml(config)
    assert isinstance(
        tuple_,
        tuple,
    ), "Problem generating tuple from config file: {}".format(config)
    obj_ = generate_object_from_tuple(tuple_)
    assert isinstance(obj_, object), "Problem generating object from tuple: {}".format(
        tuple_,
    )
    return obj_


def create_config_dict(config: Union[str, Path]) -> list:
    big = {}
    assert Path(config).exists(), "Config file does not exist: {}".format(config)
    if Path(config).is_file():
        big[config] = parse_config(config)
    elif Path(config).is_dir():
        for file in tqdm(Path(config).rglob("*.yml"), desc="Parsing Config Files"):
            big[file] = parse_config(file)
    else:
        raise ValueError("Config must be a file or directory. It is neither.")
    return big


def make_dict_list_from_tuple_list(tuple_list):
    dict_list = []
    for tup in tuple_list:
        dict_list.append({"name": tup[0], "params": tup[1]})
    return dict_list


def dump_dict_list_to_yaml(dict_list, folder):
    fullpaths = []
    for entry in dict_list:
        filename = my_hash(entry)
        full_path = Path(folder, filename + ".yaml")
        config_dict = {"inputs": {"config": entry}}
        with open(full_path, "w") as f:
            yaml.dump(config_dict, f)
        fullpaths.append(full_path.name)
    return fullpaths


def dump_stage_to_yaml(input_file, output_folder, prefix="big"):
    tuple_list = generate_tuple_list_from_yml(input_file)
    dict_list = make_dict_list_from_tuple_list(tuple_list)
    full_paths = dump_dict_list_to_yaml(dict_list, output_folder)
    return full_paths


def dump_all_stages_to_yaml(config):
    for key, value in config.items():
        input_file = value["input_file"]
        folder = value["folder"]
        _ = make_output_folder(value["folder"])
        config[key]["files"] = dump_stage_to_yaml(input_file, folder)
    return config


def generate_random_config(config, dvc_params):
    new_params = deepcopy(dvc_params)
    name = ""
    for key, value in config.items():
        rand_file = Path(value["folder"], str(choice(value["files"])))
        name += "_" + rand_file.name.split(".")[0]
        with open(rand_file, "r") as f:
            new_params[key] = yaml.load(f, Loader=yaml.FullLoader)
    return new_params, name


def count_possible_configs(config):
    count = 1
    for key, value in config.items():
        count *= len(value["files"])
    return count
