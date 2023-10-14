import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from pandas import DataFrame, read_csv, read_excel, Series

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

    def __init__(
        self,
        name: str = None,
        generate: DataGenerator = None,
        sample: SklearnDataSampler = None,
        sklearn_pipeline: SklearnDataPipeline = None,
        target: str = None,
    ):
        """Initialize the data object. If the data is generated, then generate the data and sample it. If the data is loaded, then load the data and sample it.

        Args:
            name (str, optional): The name of the data object. Defaults to None.
            generate (DataGenerator, optional): The data generator. Defaults to None.
            sample (SklearnDataSampler, optional): The data sampler. Defaults to None.
            sklearn_pipeline (SklearnDataPipeline, optional): The sklearn pipeline. Defaults to None.
            target (str, optional): The target column. Defaults to None.
        """
        logger.info(
            f"Instantiating {self.__class__.__name__} with name={name} and generate={generate} and sample={sample} and sklearn_pipeline={sklearn_pipeline} and target={target}",
        )
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
            self.sklearn_pipeline = (
                sklearn_pipeline
                if isinstance(sklearn_pipeline, (SklearnDataPipeline, type(None)))
                else SklearnDataPipeline(**sklearn_pipeline)
            )
        else:
            self.sklearn_pipeline = None
        self.target = target
        self.name = name if name is not None else my_hash(self)
        logger.debug(f"Instantiating Data with id: {self.get_name()}")

    def get_name(self):
        """Get the name of the data object."""
        return str(self.name)

    def __hash__(self):
        """Get the hash of the data object."""
        return int(my_hash(self), 16)

    def initialize(self):
        """Initialize the data object. If the data is generated, then generate the data and sample it. If the data is loaded, then load the data and sample it.
        :return: X_train, X_test, y_train, y_test
        """
        if self.generate is not None:
            result = self.generate()
            if len(result) == 2:
                result = self.sample(*result)
            else:
                assert len(result) == 4, f"Data is not generated: {self.name}"
        else:
            result = self.load(self.name)
            if isinstance(result, list) and len(result) == 2:
                result = self.sample(*result)
            elif isinstance(result, DataFrame) and self.target is not None:
                if not isinstance(result, DataFrame):
                    result = DataFrame(result)
                assert (
                    self.target in result
                ), f"Target {self.target} not in data with columns {result.columns}"
                y = result[self.target]
                if isinstance(result, DataFrame):
                    X = result.drop(self.target, axis=1)
                else:
                    X = result[~self.target]
                result = self.sample(X, y)
            else:
                assert len(result) == 4
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
            with open(filename, "r") as f:
                data = json.load(f)
        elif suffix in [".csv"]:
            data = read_csv(filename, delimiter=",", header=0)
        elif suffix in [".pkl", ".pickle"]:
            with open(filename, "rb") as f:
                data = pickle.load(f)
        elif suffix in [".xls", ".xlsx"]:
            data = read_excel(filename)
        else:
            raise ValueError(f"Unknown file type {suffix}")
        return data

    def save(self, data, filename):
        """Save data to a file
        :param data: DataFrame
        :param filename: str
        """
        if filename is not None:
            logger.info(f"Saving data to {filename}")
            suffix = Path(filename).suffix
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            if suffix in [".json"]:
                if isinstance(data, DataFrame):
                    data = data.to_dict(orient="records")
                elif isinstance(data, Series):
                    data = data.to_dict()
                elif isinstance(data, np.ndarray):
                    data = data.tolist()
                with open(filename, "w") as f:
                    json.dump(data, f)
            elif suffix in [".csv"]:
                DataFrame(data).to_csv(filename, index=False)
            elif suffix in [".pkl", ".pickle"]:
                with open(filename, "wb") as f:
                    pickle.dump(data, f)
            elif suffix in [".xls", ".xlsx"]:
                DataFrame(data).to_excel(filename)
            else:
                raise ValueError(f"Unknown file type {type(suffix)} for {suffix}")
            assert Path(filename).exists()

    def __call__(
        self,
        data_file=None,
        train_labels_file=None,
        test_labels_file=None,
    ) -> list:
        """Loads data from file if it exists, otherwise generates data and saves it to file. Returns X_train, X_test, y_train, y_test as a list of arrays, typed according to the framework.
        :param filename: str
        :return: list
        """
        result_dict = {}
        if data_file is not None and Path(data_file).exists():
            data = self.load(data_file)
            assert len(data) == 4, f"Some data is missing: {self.name}"
        else:
            data = self.initialize()
            assert len(data) == 4, f"Some data is missing: {self.name}"
            data_file = self.save(data, data_file)
        result_dict["data"] = data
        if data_file is not None:
            assert Path(data_file).exists()
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
        return data
