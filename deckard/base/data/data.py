import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union
import numpy as np
from pandas import DataFrame, read_csv, Series, read_json
from omegaconf import OmegaConf
from validators import url
from ..utils import my_hash
from .generator import DataGenerator
from .sampler import SklearnDataSampler
from .sklearn_pipeline import SklearnDataPipeline

__all__ = ["Data"]
logger = logging.getLogger(__name__)


@dataclass
class Data:
    """Data class for generating and sampling data. If the data is generated, then generate the data and sample it. When called, the data is loaded from file if it exists, otherwise it is generated and saved to file. Returns X_train, X_test, y_train, y_test as a list of arrays, typed according to the framework."""

    generate: Union[DataGenerator, None] = field(default_factory=DataGenerator)
    sample: Union[SklearnDataSampler, None] = field(default_factory=SklearnDataSampler)
    sklearn_pipeline: Union[SklearnDataPipeline, None] = field(
        default_factory=SklearnDataPipeline,
    )
    target: Union[str, None] = None
    name: Union[str, None] = None
    drop: list = field(default_factory=list)

    def __init__(
        self,
        name: str = None,
        generate: DataGenerator = None,
        sample: SklearnDataSampler = None,
        sklearn_pipeline: SklearnDataPipeline = None,
        target: str = None,
        drop: list = [],
        **kwargs,
    ):
        """Initialize the data object. If the data is generated, then generate the data and sample it. If the data is loaded, then load the data and sample it.

        Args:
            name (str, optional): The name of the data object. Defaults to None.
            generate (DataGenerator, optional): The data generator. Defaults to None.
            sample (SklearnDataSampler, optional): The data sampler. Defaults to None.
            sklearn_pipeline (SklearnDataPipeline, optional): The sklearn pipeline. Defaults to None.
            target (str, optional): The target column. Defaults to None.
        """
        if generate is not None:
            self.generate = (
                generate
                if isinstance(generate, (DataGenerator))
                else DataGenerator(**generate)
            )
        else:
            self.generate = None
        if sample is not None:
            self.sample = (
                sample
                if isinstance(sample, (SklearnDataSampler))
                else SklearnDataSampler(**sample)
            )
        else:
            self.sample = SklearnDataSampler()
        if sklearn_pipeline is not None:
            sklearn_pipeline = OmegaConf.to_container(
                OmegaConf.create(sklearn_pipeline),
            )
            self.sklearn_pipeline = (
                sklearn_pipeline
                if isinstance(sklearn_pipeline, (SklearnDataPipeline))
                else SklearnDataPipeline(**sklearn_pipeline)
            )
        else:
            self.sklearn_pipeline = None
        self.drop = drop
        self.target = target
        self.name = name if name is not None else my_hash(self)
        logger.info(f"Data initialized: {self.name}")
        logger.info(f"Data.generate: {self.generate}")
        logger.info(f"Data.sample: {self.sample}")
        logger.info(f"Data.sklearn_pipeline: {self.sklearn_pipeline}")

    def get_name(self):
        """Get the name of the data object."""
        return str(self.name)

    def __hash__(self):
        """Get the hash of the data object."""
        return int(my_hash(self), 16)

    def initialize(self, filename=None):
        """Initialize the data object. If the data is generated, then generate the data and sample it. If the data is loaded, then load the data and sample it.
        :return: X_train, X_test, y_train, y_test
        """
        if filename is not None and Path(filename).exists():
            result = self.load(filename)
        elif self.generate is not None:
            result = self.generate()
            self.save(result, self.name)
        else:
            result = self.load(self.name)
        if isinstance(result, DataFrame):
            assert self.target is not None, "Target is not specified"
            y = result[self.target]
            X = result.drop(self.target, axis=1)
            if self.drop != []:
                X = X.drop(self.drop, axis=1)
            X = X.to_numpy()
            y = y.to_numpy()
            result = [X, y]
        else:
            if self.drop != []:
                raise ValueError(
                    f"Drop is not supported for non-DataFrame data. Data is type {type(result)}",
                )
        if len(result) == 2:
            result = self.sample(*result)
        assert (
            len(result) == 4
        ), f"Data is not generated: {self.name} {result}. Length: {len(result)},"
        if self.sklearn_pipeline is not None:
            result = self.sklearn_pipeline(*result)
        return result

    def load(self, filename) -> DataFrame:
        """
        Loads data from a file
        :param filename: str
        :return: DataFrame
        """
        suffix = Path(filename).suffix
        if suffix in [".json"]:
            try:
                data = read_json(filename)
            except ValueError as e:
                logger.warning(
                    f"Error reading {filename}: {e}. Trying to read as Series.",
                )
                data = read_json(filename, typ="series")
            data = dict(data)
        elif suffix in [".csv"]:
            data = read_csv(filename, delimiter=",", header=0)
        elif suffix in [".pkl", ".pickle"]:
            with open(filename, "rb") as f:
                data = pickle.load(f)
        elif suffix in [".npz"]:
            data = np.load(filename)
        else:  # pragma: no cover
            raise ValueError(f"Unknown file type {suffix}")
        return data

    def save(self, data, filename):
        """Save data to a file
        :param data: DataFrame
        :param filename: str
        """
        if filename is not None:
            logger.debug(f"Saving data to {filename}")
            suffix = Path(filename).suffix
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, dict):
                for k, v in data.items():
                    v = str(v)
                    data[k] = v
            if suffix in [".json"]:
                if isinstance(data, (Series, DataFrame)):
                    data = data.to_dict()
                elif isinstance(data, np.ndarray):
                    data = data.tolist()
                elif isinstance(data, list):
                    new_data = []
                    for datum in data:
                        if isinstance(datum, (np.ndarray)):
                            new_data.append(datum.tolist())
                    data = new_data
                    del new_data
                elif isinstance(data, (dict, int, float, str, bool)):
                    pass
                else:  # pragma: no cover
                    raise ValueError(f"Unknown data type {type(data)} for {filename}.")
                try:
                    DataFrame(data).to_json(filename, index=False, force_ascii=False, indent=4)
                except ValueError as e:
                    if "using all scalar values" in str(e):
                        Series(data).to_json(filename, index=False, force_ascii=False, indent=4)
                    else:
                        raise e
            elif suffix in [".csv"]:
                assert isinstance(
                    data,
                    (Series, DataFrame, dict, np.ndarray),
                ), f"Data must be a Series, DataFrame, or dict, not {type(data)} to save to {filename}"
                if isinstance(data, (np.ndarray)):
                    data = DataFrame(data)
                data.to_csv(filename, index=False)
            elif suffix in [".pkl", ".pickle"]:
                with open(filename, "wb") as f:
                    pickle.dump(data, f)
            elif suffix in [".npz"]:
                np.savez(filename, data)
            else:  # pragma: no cover
                raise ValueError(f"Unknown file type {type(suffix)} for {suffix}")
            assert Path(filename).exists()

    def __call__(
        self,
        data_file=None,
        train_labels_file=None,
        test_labels_file=None,
        **kwargs,
    ) -> list:
        """Loads data from file if it exists, otherwise generates data and saves it to file. Returns X_train, X_test, y_train, y_test as a list of arrays, typed according to the framework.
        :param filename: str
        :return: list
        """
        if Path(self.name).is_file() or url(self.name):
            new_data_file = data_file
            data_file = self.name
        else:
            new_data_file = data_file
        if data_file is not None and Path(data_file).exists():
            data = self.initialize(data_file)
        elif "data" in kwargs:
            assert (
                len(kwargs["data"]) == 4
            ), f"Data must be length 4, not {len(kwargs['data'])}"
            data = kwargs["data"]
        else:
            data = self.initialize()
        if train_labels_file is not None:
            self.save(data[2], train_labels_file)
            assert Path(
                train_labels_file,
            ).exists(), f"Error saving train labels to {train_labels_file}"
        if test_labels_file is not None:
            self.save(data[3], test_labels_file)
            assert Path(
                test_labels_file,
            ).exists(), f"Error saving test labels to {test_labels_file}"
        if new_data_file is not None:
            self.save(data, new_data_file)
        return data
