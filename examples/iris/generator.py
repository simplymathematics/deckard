import logging
from pathlib import Path
from typing import Union
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
from deckard.base.hashable import BaseHashable, my_hash

# Scikit-learn:
from sklearn.model_selection import ParameterGrid
import dvc.api

SUPPORTED_defenceS = (Postprocessor, Preprocessor, Transformer, Trainer)
SUPPORTED_MODELS = (
    PyTorchClassifier,
    ScikitlearnEstimator,
    KerasClassifier,
    TensorFlowClassifier,
)
logger = logging.getLogger(__name__)

# yaml.add_constructor("!!python/object/apply", lambda loader, node: loader.construct_scalar(node))


class Generator(BaseHashable):
    def __init__(
        self,
        root_folder: Union[str, Path],
        prototype_file: Union[str, Path],
        config_regex: str,
        config_path: Union[str, Path],
        queue_file: Union[str, Path],
        main_file: Union[str, Path] = "authoritative.csv",
    ):
        """
        Initializes experiment generator from root_folder, config_path, regular expression for config files, a prototype pipeline file, and a prototype params file.
        :params root_folder: the root folder to start the experiment generator from
        :params config_regex: the regular expression to use for finding config files
        :params config_path: the path to the config files
        :params prototype_file: the prototype pipeline file
        :params queue_file: the queue file
        :params main_file: the main queue file. Not intended to be overwritten.
        """

        self.root_folder = root_folder
        self.config_path = config_path
        self.config_regex = config_regex
        self.proto_file = Path(self.root_folder, prototype_file)
        self.pipeline, self.params = self.parse_prototype(self.proto_file)
        self.output = Path(self.root_folder, self.config_path)
        layers = self.pipeline.keys()
        queue = {}
        for layer in layers:
            config = Path(root_folder, config_path, layer, config_regex)
            if not config.is_file():
                raise FileNotFoundError(f"Config file {config} not found.")
            with config.open("r") as f:
                entries = yaml.load(f, Loader=yaml.FullLoader)
            if isinstance(entries, list):
                exp_list = []
                for entry in entries:
                    exp_list.extend(self.parse_entry(entry))
            elif isinstance(entries, dict):
                exp_list = self.parse_entry(entries)
            else:
                raise ValueError("Contend of config file must be a list or a dict.")
            ids_ = []
            Path(self.root_folder, layer).mkdir(parents=True, exist_ok=True)
            for id_, name, sub_params in exp_list:
                with open(Path(self.root_folder, layer, id_ + ".yaml"), "w") as f:
                    dict_ = {"name": name}
                    dict_.update(**sub_params)
                    yaml.dump({layer + "_params": dict_}, f)
                ids_.append(id_)
            queue[layer] = ids_
        queue = list(ParameterGrid(queue))
        self.queue = queue
        self.queue_file = Path(queue_file)
        self.main_file = Path(main_file)

    # def parse_list(self, filename: str) -> list:
    #     """
    #     Parses a yml file, generates a an exhaustive list of parameter combinations for each entry in the list, and returns a single list of tuples.
    #     """
    #     full_list = []

    #     if Path(filename).is_file():
    #         with open(filename, "r") as stream:
    #             yml_list = yaml.load(stream, Loader = yaml.FullLoader)
    #     else:
    #         raise FileNotFoundError(f"File {filename} not found.")
    #     # check that featurizers is a list
    #     for entry in yml_list:
    #         print(entry)
    #         input("Press Enter to continue...")
    #         full_list.extend(self.parse_entry(**entry))
    #     return full_list

    def parse_entry(self, entry: dict) -> list:
        """
        Parses a yml file and returns a dictionary.
        """
        full_list = []
        special_keys = {}

        for key, value in entry.items():
            if isinstance(value, (tuple, float, int, str)):
                special_values = value
                special_key = key
                special_keys[special_key] = special_values
        for key in special_keys.keys():
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

    def parse_prototype(
        self, filename: Union[str, Path], vars_dict: dict = None,
    ) -> dict:
        with open(filename, "r") as f:
            pipeline = yaml.load(f, Loader=yaml.FullLoader)["stages"]
        with open(filename, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)["vars"]
        return pipeline, params

    def parse_pipe(self):
        layers = self.queue[0]
        replace = {list(x.keys())[0]: list(x.values())[0] for x in self.params}
        for layer in layers:
            id_ = layers[layer]
            replace[layer + "_folder"] = str(
                Path(self.root_folder, layer, id_).relative_to(self.root_folder),
            )
            replace[layer + "_params"] = (
                str(
                    Path("..", self.output, layer, id_ + ".yaml").relative_to(
                        self.output,
                    ),
                )
                + f":{layer}_params"
            )
        long_string = str(self.pipeline)
        for key in replace:
            value = str(replace[key])
            key = "${" + key + "}"
            long_string = long_string.replace(key, value)
        parsed = eval(long_string)
        return parsed

    def __iter__(self):
        for item in self.queue:
            yield item

    def __len__(self):
        return len(self.queue)

    def generate_param_file(self, path=None, working_directory=None):
        path_list = []
        pipeline = self.parse_pipe()
        new_path = Path(working_directory, path, my_hash(pipeline))
        new_path.mkdir(parents=True, exist_ok=True)
        with open(Path(new_path, "params.yaml"), "w") as f:
            yaml.dump({"stages": pipeline}, f)
        path_list.append(str(Path(new_path, "dvc.yaml").relative_to(working_directory)))
        return path_list

    def generate_dvc_file(self, path=None, working_directory=None):
        path_list = []
        pipeline = self.parse_pipe()
        new_path = Path(working_directory, path, my_hash(pipeline))
        new_path.mkdir(parents=True, exist_ok=True)
        with open(Path(new_path, "dvc.yaml"), "w") as f:
            yaml.dump({"stages": pipeline}, f)
        path_list.append(str(Path(new_path, "dvc.yaml").relative_to(working_directory)))
        return path_list

    def __call__(self, path=None, working_directory=None, sort_by=None):
        path_list = []
        param_files = []
        queue = pd.DataFrame(list(self.queue))
        while len(self) > 0:
            path_list.extend(self.generate_dvc_file(path, working_directory))
            param_files.extend(self.generate_param_file(path, working_directory))
            self.queue.pop(0)
        queue["pipeline"] = path_list
        queue["params"] = param_files
        if sort_by is not None:
            pd.DataFrame(queue).sort_values(by=sort_by).to_csv(
                Path(self.output, path, self.queue_file), index=False,
            )
            pd.DataFrame(queue).sort_values(by=sort_by).to_csv(
                Path(self.output, path, self.main_file), index=False,
            )
        else:
            pd.DataFrame(queue).to_csv(Path(self.output, self.queue_file), index=False)
            pd.DataFrame(queue).to_csv(Path(self.output, self.main_file), index=False)
        with open(Path(self.root_folder, "params.yaml"), "w") as f:
            # yaml.dump({"param_files" : param_files}, f)
            yaml.dump({"pipelines": path_list, "param_files": param_files}, f)

        return path_list


if __name__ == "__main__":
    conf_file = "default.yaml"
    root_path = Path("../../examples/iris").resolve()
    config_path = "configs"
    stage = "prepare"
    param = stage + "_params"
    gen = Generator(
        root_folder=root_path,
        config_path=config_path,
        config_regex=conf_file,
        prototype_file="dvc.yaml",
        queue_file="queue.csv",
    )
    assert len(gen) > 0
    pipeline = gen(path="queue", working_directory=Path(root_path))
    params = dvc.api.params_show(Path(root_path, pipeline[0]))["stages"]
    assert "$" not in str(params), "There are unresolved variables in the pipeline"
