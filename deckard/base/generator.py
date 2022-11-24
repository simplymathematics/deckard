# Standard Library:
import json
import logging
import os

import pandas as pd
import yaml
from art.defences.postprocessor import Postprocessor
from art.defences.preprocessor import Preprocessor
from art.defences.trainer import Trainer
from art.defences.transformer import Transformer
from art.estimators import ScikitlearnEstimator

# Adversarial Robustness Toolbox
from art.estimators.classification import (
    KerasClassifier,
    PyTorchClassifier,
    TensorFlowClassifier,
)

# Scikit-learn:
from sklearn.model_selection import ParameterGrid
from pathlib import Path
from hashable import BaseHashable, my_hash

SUPPORTED_DEFENSES = (Postprocessor, Preprocessor, Transformer, Trainer)
SUPPORTED_MODELS = (
    PyTorchClassifier,
    ScikitlearnEstimator,
    KerasClassifier,
    TensorFlowClassifier,
)
logger = logging.getLogger(__name__)


class Generator(BaseHashable):
    def __init__(
        self,
        config_file: str,
        root_path: str,
        result_path: str,
        config_path: str = "configs",
        **kwargs,
    ):
        """
        Initializes experiment generator from a config file and path.
        :param config_file: the config file to read from
        :param root_path: the path to the config file
        :param result_path: the path to the result file. Will create
        """
        root_path = os.path.abspath(root_path)
        assert os.path.isdir(root_path), f"{root_path} is not a directory"
        os.chdir(root_path)
        self.path = root_path
        self.params_path = config_path
        self.params_file = os.path.join(config_path, config_file)
        self.input = os.path.join(self.path, config_path, config_file)
        if not os.path.isfile(self.input):
            raise ValueError(str(self.input) + " file does not exist")
        self.output = os.path.join(self.path, result_path)
        if not os.path.isdir(self.output):
            os.mkdir(self.output)
        self.params = self.set_config()
        self.list = self.generate_tuple_list_from_yml(self.params[1], **kwargs)

    def __call__(self, filename: str):
        """
        Generates a json file for a given experiment.
        :param filename: the name of the json file
        """
        assert isinstance(filename, str), "Filename {} must be a string.".format(
            filename,
        )
        assert os.path.isdir(self.output), f"{self.output} is not a directory"
        # check if the file exists
        if not os.path.isfile(str(filename)):
            paths = self.generate_experiment_list(filename)
        else:
            raise ValueError(str(filename) + " file already exists")
        return paths

    def set_config(self) -> dict:
        """
        Finds all config files in the current directory and returns a list of config files.
        :return config: a dict of config files
        """
        name = self.input.split(".")[0]
        file = self.input
        config = (name, file)
        return config
    def generate_json(self, path: str, filename: str, name: str, params: dict):
        """
        Generates a json file for a given experiment.
        :param path: the path to save the json file to
        :param filename: the name of the json file
        :param params: the parameters to use for the experiment
        """
        assert isinstance(path, str), "path {} must be a string.".format(path)
        assert isinstance(filename, str), "Filename {} must be a string.".format(
            filename,
        )
        assert isinstance(params, dict), "Params {} must be a dict.".format(params)
        assert os.path.isdir(path), f"{path} is not a directory"
        dictionary = {}
        dictionary["name"] = name
        dictionary["params"] = params
        with open(os.path.join(path, filename + ".json"), "w") as f:
            json.dump(dictionary, f)

    def generate_yml(self, path: str, filename: str, name: str, params: dict):
        """
        Generates a yml file for a given experiment.
        :param path: the path to save the yml file to
        :param name: the name of the experiment
        :param params: the parameters to use for the experiment
        """
        assert isinstance(path, str), "path {} must be a string.".format(path)
        assert isinstance(filename, str), "Name {} must be a string.".format(filename)
        assert isinstance(params, dict), "Params {} must be a dict.".format(params)
        assert os.path.isdir(path), f"{path} is not a directory"
        dictionary = {}
        dictionary["name"] = name
        dictionary["params"] = params
        with open(os.path.join(path, filename), "w") as f:
            yaml.dump(dictionary, f)

    def generate_directory_tree(self, filename="params"):
        paths = []
        for entry in self.list:
            name = entry[0]  # drop this in lieu of hash below since this is
            params = entry[1]
            path = my_hash(entry)
            path = os.path.join(self.output, path)
            if not os.path.isdir(path):
                os.makedirs(path)
            if not os.path.isdir(os.path.join(path, "configs")):
                foo = os.path.join(path, "configs")
                os.makedirs(foo)
                assert os.path.isdir(foo), "Your error is here."
                # shutil.copytree(os.path.join(self.path, self.params_path), foo)
            self.generate_yml(
                path=path,
                filename=self.params_file.split(os.sep)[-1],
                name=name,
                params=params,
            )
            self.generate_json(path=path, filename=filename, name=name, params=params)
            paths.append(path)
        return paths

    def generate_experiment_list(self, filename):
        """
        Generates a list of experiments.
        :param path: the path to save the yml file to
        :param name: the name of the experiment
        :param params: the parameters to use for the experiment
        """

        assert isinstance(filename, str), "Name {} must be a string.".format(filename)
        paths = self.generate_directory_tree()
        names = [path.split(os.sep)[-1] for path in paths]
        df = pd.DataFrame(self.list)
        df["ID"] = names
        df.columns = ["object_type", "params", "ID"]
        filename = os.path.join(self.output, filename + ".csv")
        if os.path.isfile(filename):
            df.to_csv(filename, mode="a", header=False)
        else:
            df.to_csv(filename, mode="w", header=True)
        assert len(df) == len(paths), "Error generating experiment list"
        return df

def parse_entry(self, entry:dict) -> list:
        """
        Parses a yml file and returns a dictionary.
        """
        full_list = []
        special_keys = {}
        # print(entry)
        # input("Press Enter to continue...")
        for key, value in entry.items():
            if isinstance(value, (tuple, float, int, str)):
                special_keys[key] = value
                entry.pop(key)       
        grid = ParameterGrid(entry)
        
        for combination in grid:
            if "special_keys" in locals():
                for key, value in special_keys.items():
                    combination[key] = value
            name = combination.pop("name")
            hash_ = my_hash({"name": name, "params": combination})
            full_list.append((hash_, name, combination))
        return full_list
    
def generate_tuple_list_from_yml(filename: str) -> list:
    """
    Parses a yml file, generates a an exhaustive list of parameter combinations for each entry in the list, and returns a single list of tuples.
    """
    assert isinstance(filename, str), "Filename {} must be a string.".format(
        filename,
    )
    assert os.path.isfile(filename), f"{filename} does not exist"
    full_list = []
    LOADER = yaml.FullLoader
    # check if the file exists
    if not os.path.isfile(str(filename)):
        raise ValueError(str(filename) + " file does not exist")
    # read the yml file
    try:
        with open(filename, "r") as stream:
            yml_list = yaml.load(stream, Loader=LOADER)
    except yaml.YAMLError as exc:
        logger.error(exc)
        raise ValueError("Error parsing yml file {}".format(filename))
    # check that featurizers is a list
    for entry in yml_list:
        if not isinstance(entry, dict):
            raise ValueError("Error parsing yml file {}".format(filename))
        
        full_list.extend(parse_entry(config))
    return full_list

    
    
    

if __name__ == '__main__':
    full_list = generate_tuple_list_from_yml("configs/data.yaml")
    print(full_list)

