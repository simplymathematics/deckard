import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple, Union, Literal
from collections import OrderedDict

import hydra
import numpy as np
from hydra.utils import instantiate, call
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from deckard.base.hashable import my_hash
from deckard.layers.parse import parse
import pandas as pd
import yaml
@dataclass
class SamplerConfig:
    random_state: int = 42
    shuffle: bool = True
    stratify: bool = True
    test_size : int = 1000
    time_series: bool = False
    train_size : int = 1000
    
    def __call__(self, X, y, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = asdict(self)
        time_series = params.pop("time_series", False)
        stratify = params.pop("stratify", False)
        if stratify is True:
            stratify = y
        else:
            stratify = None
        if time_series is False:
            for k in kwargs:
                if k in params:
                    raise ValueError(f"Parameter {k} already set.")
                else:
                    params[k] = kwargs[k]
            results = train_test_split(X, y, **params)
        else:
            raise NotImplementedError("Time series not implemented yet.")
        return results

@dataclass
class ClassificationGeneratorConfig:
    n_samples: int = 10000
    n_features: int = 100
    n_informative: int = 99
    n_redundant: int = 0
    n_repeated: int = 0
    n_classes: int = 2
    n_clusters_per_class: int = 1
    weights: Union[list, None] = None
    class_sep: float = 1.0
    hypercube: bool = True
    shift: float = 0.0
    scale: float = 1.0
    shuffle: bool = True
    random_state: int = 42
    
    def __call__(self) -> Any:
        params = asdict(self)
        return make_classification(**params)

@dataclass
class RegressionGeneratorConfig():
    n_samples: int = 100
    n_features: int = 100
    n_informative: int = 2
    n_targets: int = 1
    bias: float = 0.0
    effective_rank: Union[int, None] = None
    tail_strength: float = 0.5
    noise: float = 0.0
    shuffle: bool = True
    coef: bool = False
    random_state: int = None
    
    def __call__(self) -> Any:
        params = asdict(self)
        return make_regression(**params)
    
@dataclass
class LoaderConfig:
    directory : Union[str, Path] = "data"
    regex : Union[str, Path] = "*"
    suffix : Union[str, Path] = ".npz"
    random_state: int = 42
    kwargs : Dict = field(default_factory=dict)
    
    def _load(self, *args, **kwargs):
        if self.suffix == ".npz":
            return np.load(*args, **kwargs)
        elif self.suffix == ".csv":
            yield pd.read_csv(*args, **kwargs)
        elif self.suffix == ".json":
            yield pd.read_json(*args, **kwargs)
        elif self.suffix == ".pkl" or self.suffix == ".pickle":
            yield pd.read_pickle(*args, **kwargs)
        elif self.suffix == ".yaml":
            with open(*args, "r") as f:
                params = yaml.safe_load(f)
            yield pd.DataFrame(params, **kwargs)
        elif self.suffix in [".h5", ".hdf5"]:
            yield pd.read_hdf(*args, **kwargs)
        elif self.suffix == ".feather":
            yield pd.read_feather(*args, **kwargs)
        elif self.suffix == ".parquet":
            yield pd.read_parquet(*args, **kwargs)
        elif self.suffix in ['.xls', '.xlsx']:
            yield pd.read_excel(*args, **kwargs)
        elif self.suffix in ["txt", ".tsv", "md"]:
            yield pd.read_table(*args, **kwargs)
        elif self.suffix == ".html":
            yield pd.read_html(*args, **kwargs)
        elif self.suffix == ".sql":
            yield pd.read_sql(*args, **kwargs)
        elif self.suffix == "sqlalchemy":
            yield pd.read_sql_table(*args, **kwargs)
        elif self.suffix == ".gbq":
            yield pd.read_gbq(*args, **kwargs)
        elif self.suffix in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif']:
            from PIL import Image
            df = pd.DataFrame()
            for image in Path(*args).glob(**kwargs):
                
                df[str(Path(image))] = Image.open(image)
            yield df
        elif self.suffix in ['.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a', '.wma']:
            from pydub import AudioSegment
            df = pd.DataFrame()
            for audio in Path(*args).glob(**kwargs):
                df[str(Path(audio))] = AudioSegment.from_file(audio)
            yield df
        elif self.suffix in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.mpg', '.mpeg']:
            import cv2
            df = pd.DataFrame()
            for video in Path(*args).glob(**kwargs):
                df[str(Path(video))] = cv2.VideoCapture(video)
            yield df
        elif self.suffix in ['.zip', '.tar', '.gz', '.bz2', '.7z', '.rar']:
            import zipfile
            df = pd.DataFrame()
            for archive in Path(*args).glob(**kwargs):
                df[str(Path(archive))] = zipfile.ZipFile(archive)
            yield df
        else:
            raise NotImplementedError(f"Suffix {self.suffix} not implemented. Try adding it to the LoaderConfig._load method in this file: {__file__}")
    
        def __call__(self, output_file:Union[str, Path], output_path:Union[str, Path]=None, output_suffix:Union[str,Path]) -> Any:
            df = pd.DataFrame()
            for data in self:
                df[my_hash(data)] = data
            if output_path is None:
                output_path = self.directory
            if output_suffix is None:
                output_suffix = self.suffix
            if output_file is None:
                output_file = my_hash(df)
            
        def __iter__(self):
            kwargs = asdict(self.kwargs)
            random.seed(kwargs.pop("random_state"))
            new_list = list(Path(kwargs.pop("directory")).glob(kwargs.pop("regex") + kwargs.pop("suffix")))
            shuffle = kwargs.pop("shuffle", True)
            if shuffle is True:
                random.shuffle(new_list)
            for f in new_list:
                yield self._load(f, **kwargs)
                
        def _save(self, *args, **kwargs):
            if self.suffix == ".npz":
                np.savez(*args, **kwargs)
            elif self.suffix == ".csv":
                pd.DataFrame(*args, **kwargs).to_csv()
            elif self.suffix == ".json":
                pd.DataFrame(*args, **kwargs).to_json()
            elif self.suffix == ".pkl" or self.suffix == ".pickle":
                pd.DataFrame(*args, **kwargs).to_pickle()
            elif self.suffix == ".yaml":
                with open(*args, "w") as f:
                    yaml.safe_dump(kwargs, f)
            elif self.suffix in [".h5", ".hdf5"]:
                pd.DataFrame(*args, **kwargs).to_hdf()
            elif self.suffix == ".feather":
                pd.DataFrame(*args, **kwargs).to_feather()
            elif self.suffix == ".parquet":
                pd.DataFrame(*args, **kwargs).to_parquet()
            elif self.suffix in ['.xls', '.xlsx']:
                pd.DataFrame(*args, **kwargs).to_excel()
            elif self.suffix in ["txt", ".tsv", "md"]:
                pd.DataFrame(*args, **kwargs).to_table()
            elif self.suffix == ".html":
                pd.DataFrame(*args, **kwargs).to_html()
            elif self.suffix == ".sql":
                pd.DataFrame(*args, **kwargs).to_sql()
            elif self.suffix == "sqlalchemy":
                pd.DataFrame(*args, **kwargs).to_sql_table()
            elif self.suffix == ".gbq":
                pd.DataFrame(*args, **kwargs).to_gbq()
            elif self.suffix in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif']:
                from pil import Image
                df = pd.DataFrame()
                for image in self:
                    image = Image.fromarray(image)
                    filename = my_hash(image) + self.suffix
                    path = self.directory
                    full_path = str(Path(path) / filename)
                    image.save(full_path)
                    df[my_hash(image)] = full_path
                return df
            elif self.suffix in ['.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a', '.wma']:
                from pydub import AudioSegment
                for audio in self:
                    filename = my_hash(audio) + self.suffix
                    path = self.directory
                    full_path = str(Path(path) / filename)
                    audio.export(full_path)
                    df[my_hash(audio)] = full_path
                return df
            elif self.suffix in ['.zip', '.tar', '.gz', '.bz2', '.7z', '.rar']:
                import zipfile
                df = pd.DataFrame()
                for archive in self:
                    filename = my_hash(archive) + self.suffix
                    path = self.directory
                    full_path = str(Path(path) / filename)
                    archive.export(full_path)
                    df[my_hash(archive)] = full_path
                return df
            elif self.suffix in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.mpg', '.mpeg']:
                import cv2
                df = pd.DataFrame()
                for video in self:
                    filename = my_hash(video) + self.suffix
                    path = self.directory
                    full_path = str(Path(path) / filename)
                    video.export(full_path)
                    df[my_hash(video)] = full_path
                return df
            else:
                raise NotImplementedError(f"Suffix {self.suffix} not implemented. Try adding it to the LoaderConfig._save method in this file: {__file__}")
        
    class Data():
        yaml_tag = "!deckard.Data"
        def __init__(self, name, **kwargs) -> None:
            if Path(name).is_dir():
                self.directory = name
                assert "regex" in kwargs, "Must specify regex if directory is specified."
                self.regex = kwargs.pop("regex")
                assert "suffix" in kwargs, "Must specify suffix if directory is specified."
                self.suffix = kwargs.pop("suffix")
                self.name = f"{self.directory}/{self.regex}{self.suffix}"
                kwargs.pop("name", None)
            elif Path(name).is_file():
                self.directory = kwargs.pop("directory", Path(name).parent)
                self.regex = kwargs.pop("regex", Path(name).stem)
                self.suffix = kwargs.pop("suffix", Path(name).suffix)
                self.name = name
            elif isinstance(name, str):
                self.directory = kwargs.pop("directory", "data")
                self.regex = kwargs.pop("regex", name)
                self.suffix = kwargs.pop("suffix", ".npz")
                self.name = f"{self.directory}/{self.regex}{self.suffix}"
            else:
                raise ValueError(f"Cannot parse {name} as a path.")
            if "generator" in kwargs:
                self.generator = field(default_factory=Union[ClassificationGeneratorConfig, RegressionGeneratorConfig])
            if "loader" in kwargs:
                self.loader = field(default_factory=LoaderConfig)
            # if "sklearn_pipeline" in kwargs:
            #     self.sklearn_pipeline = field(default_factory=SklearnPipelineDataConfig)
            # if "torch_pipeline" in kwargs:
            #     self.torch_pipeline = field(default_factory=TorchPipelineDataConfig)
            # if "keras_pipeline" in kwargs:
            #     self.keras_pipeline = field(default_factory=KerasPipelineDataConfig)
            # if "transform" in kwargs:
            #     self.transform = field(default_factory=TransformConfig)
            if "sampler" in kwargs:
                self.sampler = field(default_factory=SamplerConfig)
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    setattr(self, k, instantiate(v))
                    kwargs.pop(k)
            
        @classmethod
        def to_yaml(cls, representer, node):
            string_ = f"{node.name}"
            for node in fields(node):
                if node.name not in ["name", "directory", "regex", "suffix"]:
                    string_ += f"{node.name}: {node.value}"
            return representer.represent_scalar(cls.yaml_tag, string_)
        
        @classmethod
        def from_yaml(cls, constructor, node):
            kwargs = {}
            for nodes in fields(node):
                if nodes.name == "name":
                    name = nodes.value
                else:
                    kwargs[nodes.name] = nodes.value
            return cls(name, **kwargs)
                    
        
        
        def __repr__(self) -> str:
            params = vars(self)
            del params['name']
            return f"{self.__class__.__name__}({self.name}, {params})"
        
    class FactoryConfig:
        def __init__(self, **kwargs):
            if "name" or "__target__" in kwargs:
                assert not ("name" in kwargs and "__target__" in kwargs), "Cannot specify both name and __target__."
                self.__target__ = kwargs.pop("name", kwargs.pop("__target__"))
            else:
                raise ValueError("Must specify name or __target__.")
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    setattr(self, k, instantiate(v))
                    kwargs.pop(k)
                elif isinstance(v, list):
                    setattr(self, k, [instantiate(i) if isinstance(i, dict) else i for i in v])
                    kwargs.pop(k)
                elif isinstance(v, (int, float, str, bool)):
                    setattr(self, k, v)
                    kwargs.pop(k)
                else:
                    raise NotImplementedError(f"Cannot parse {v} of type {type(v)}. Please specify a dict, list, int, float, str, or bool. You could also modify the FactoryConfig class in this file: {__file__}")
        
        def __iter__(self):
            list_of_vars = flatten_dict(vars(self))
            big_grid = ParameterGrid(list_of_vars)
            return unflatten_dict(next(big_grid))
        
        def __call__(self):
            while len(self) > 0:
                yield self.pop()
        
        def __repr__(self) -> str:
            params = vars(self)
            del params['__target__']
            return f"{self.__class__.__name__}({self.__target__}, {params})"
        
        def __len__(self):
            return len(ParameterGrid(flatten_dict(vars(self))))
        
        
    
    class Model:
        def __init__(self, name, **kwargs) -> None:
            if Path(name).is_dir():
                self.directory = name
                assert "regex" in kwargs, "Must specify regex if directory is specified."
                self.regex = kwargs.pop("regex")
                assert "suffix" in kwargs, "Must specify suffix if directory is specified."
                self.suffix = kwargs.pop("suffix")
                self.name = f"{self.directory}/{self.regex}{self.suffix}"
                kwargs.pop("name", None)
            elif Path(name).is_file():
                self.directory = kwargs.pop("directory", Path(name).parent)
                self.regex = kwargs.pop("regex", Path(name).stem)
                self.suffix = kwargs.pop("suffix", Path(name).suffix)
                self.name = name
            elif isinstance(name, str):
                self.directory = kwargs.pop("directory", "data")
                self.regex = kwargs.pop("regex", name)
                self.suffix = kwargs.pop("suffix", ".npz")
                self.name = f"{self.directory}/{self.regex}{self.suffix}"
            else:
                raise ValueError(f"Cannot parse {name} as a path.")
            if "sklearn_pipeline" in kwargs:
            #     self.sklearn_pipeline = field(default_factory=SklearnPipelineModelConfig)
            #     kwargs.pop("sklearn_pipeline")
            # if "torch_pipeline" in kwargs:
            #     self.torch_pipeline = field(default_factory=TorchPipelineModelConfig)
            #     kwargs.pop("torch_pipeline")
            # if "keras_pipeline" in kwargs:
            #     self.keras_pipeline = field(default_factory=KerasPipelineModelConfig)
            #     kwargs.pop("keras_pipeline")
            # if "art_pipeline" in kwargs:
            #     self.art_pipeline = field(default_factory=ArtPipelineModelConfig)
            #     kwargs.pop("art_pipeline")
            
            
        


    
    
# @dataclass
# class DataConfig:
#     name : Union[str, None] = "classification"
#     generate : Union[ClassificationGeneratorConfig, None] = field(default_factory=ClassificationGeneratorConfig)
#     sample : Union[SamplerConfig, None] = field(default_factory=SamplerConfig)
#     filename : Union[str, None] = None
#     path : Union[str, None] = "data"
#     filetype : Union[str, None] = "npz"
    
#     def _initialize(self, filename=None, path=None, filetype =None) -> Tuple:
#         if filename is None:
#             filename = self.filename if self.filename is not None else my_hash(self)
#         if path is None:
#             path = self.path if self.path is not None else hydra.utils.get_original_cwd()
#         if filetype is None:
#             filetype = self.filetype if self.filetype is not None else "npz"
#         full_path =Path(path, str(filename)+ "." + filetype)
      
#         if not full_path.is_file():
#             if Path(self.name).is_file():
#                 X_train, X_test, y_train, y_test = self._load(filename=self.name, path=self.path, filetype=self.filetype)
#             elif self.generate is not None:
#                 generator = instantiate(self.generate)
#                 generator.pop("__target__", None) if isinstance(generator, dict) else None
#                 generator = ClassificationGeneratorConfig(**generator)
#                 X, y = generator()
#             if self.sample is not None:
#                 sampler = instantiate(self.sample)
#                 sampler.pop("__target__", None) if isinstance(sampler, dict) else None
#                 sampler = SamplerConfig(**sampler)
#                 X_train, X_test, y_train, y_test = self.sample(X = X, y = y)
#             else:
#                 raise NotImplementedError("No data sampling method specified.")
#             self._save(X_train, X_test, y_train, y_test)
#         else:
#             X_train, X_test, y_train, y_test = self._load(filename=filename, path=path, filetype=filetype)
#         return X_train, X_test, y_train, y_test
    
#     def _load(self, filename=None, path=None, filetype=None) -> Tuple:
#         if path is None:
#             path = self.path if self.path is not None else hydra.utils.get_original_cwd()
#             path = Path(path)
#         if filename is None:
#             filename = self.filename if self.filename is not None else my_hash(self)
#         if filetype is None:
#             filetype = self.filetype if self.filetype is not None else "npz"
#         path = path / Path(str(filename) + "." + filetype)
#         data = np.load(str(path.resolve()))
#         X_train = data["X_train"]
#         X_test = data["X_test"]
#         y_train = data["y_train"]
#         y_test = data["y_test"]
#         return X_train, X_test, y_train, y_test
    
#     def _save(self, X_train, X_test, y_train, y_test, filename=None, path=None, filetype=None) -> None:
#         if path is None:
#             path = self.path if self.path is not None else hydra.utils.get_original_cwd()
#         Path(path).mkdir(parents=True, exist_ok=True)
#         if filename is None:
#             filename = self.filename if self.filename is not None else my_hash(self)
#         if filetype is None:
#             filetype = self.filetype if self.filetype is not None else "npz"
#         path = Path(path) / Path(str(filename) + "." + filetype)
#         if filetype == "npz":
#             np.savez(file=str(path.resolve()), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
#         else:
#             raise NotImplementedErrror(f"Filetype {filetype} not implemented yet.")
        
#         return path

#     def __call__(self, filename=None, path=None, filetype=None) -> Tuple:
#         X_train, X_test, y_train, y_test = self._initialize(filename=filename, path=path, filetype=filetype)
#         file = self._save(X_train, X_test, y_train, y_test, filename=filename, path=path, filetype=filetype)
#         assert Path(file).exists(), f"File {file} does not exist."
#         return X_train, X_test, y_train, y_test
    
@hydra.main(
    version_base=None,
    config_path=str(Path(os.getcwd(), "conf-tmp")),
    config_name="config",
)
def my_app(cfg) -> None:
    import tempfile
    import uuid
    yaml_config = dict(cfg)
    parsed_config = parse(yaml_config)
    instance = instantiate(parsed_config)
    generator_params = instantiate(instance.data.generate)
    generator = generator_params.pop("__target__", "ClassificationGeneratorConfig")
    generator = generator(**generator_params)
    sampler = instantiate(instance.data.sample)
    sampler.pop("__target__")
    sampler = SamplerConfig(**sampler)
    data = DataConfig(name=instance.data.name, generate=generator, sample=sampler, filename =uuid.uuid4(), path=tempfile.gettempdir(), filetype="npz")
    X_train, X_test, y_train, y_test = data()
    assert isinstance(X_train, np.ndarray), f"X_train is not a numpy array."
    assert isinstance(X_test, np.ndarray), f"X_test is not a numpy array."
    assert isinstance(y_train, np.ndarray), f"y_train is not a numpy array."
    assert isinstance(y_test, np.ndarray), f"y_test is not a numpy array."
if __name__ == "__main__":
    
    parsed_config = my_app()
    