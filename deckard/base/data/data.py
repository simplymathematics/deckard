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
from .sampler import SklearnSplitSampler
from .sklearn_pipeline import SklearnDataPipeline

__all__ = ["Data"]
logger = logging.getLogger(__name__)


@dataclass
class Data:
    """Data class for generating and sampling data. If the data is generated, then generate the data and sample it. When called, the data is loaded from file if it exists, otherwise it is generated and saved to file. Returns X_train, X_test, y_train, y_test as a list of arrays, typed according to the framework."""

    generate: Union[DataGenerator, None] = field(default_factory=DataGenerator)
    sample: Union[SklearnSplitSampler, None] = field(
        default_factory=SklearnSplitSampler,
    )
    sklearn_pipeline: Union[SklearnDataPipeline, None] = field(
        default_factory=SklearnDataPipeline,
    )
    target: Union[str, None] = None
    name: Union[str, None] = None
    drop: list = field(default_factory=list)
    alias: Union[str, None] = None

    def __init__(
        self,
        name: str = None,
        generate: DataGenerator = None,
        sample: SklearnSplitSampler = None,
        sklearn_pipeline: SklearnDataPipeline = None,
        target: str = None,
        drop: list = [],
        alias: str = None,
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
            sample = OmegaConf.to_container(OmegaConf.create(sample), resolve=True)
            self.sample = (
                sample
                if isinstance(sample, (SklearnSplitSampler))
                else SklearnSplitSampler(**sample)
            )
        else:
            self.sample = SklearnSplitSampler()
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
        self.alias = alias
        logger.debug(f"Data initialized: {self.name}")
        logger.debug(f"Data.generate: {self.generate}")
        logger.debug(f"Data.sample: {self.sample}")
        logger.debug(f"Data.sklearn_pipeline: {self.sklearn_pipeline}")

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
        if filename is not None and Path(filename).exists() or url(filename):
            logger.info(f"Loading data from {filename}")
            result = self.load(filename)
        elif self.generate is not None:
            logger.info(f"Generating data for {self.generate}")
            result = self.generate()
            self.save(result, filename)
        else:
            logger.info(f"Loading data from {self.name}")
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
        if len(result) == 2:
            result = self.sample(*result)
        else:
            if self.drop != []:
                raise ValueError(
                    f"Drop is not supported for non-DataFrame data. Data is type {type(result)}",
                )
            assert (
                len(result) == 4
            ), f"Data is not generated: {self.name} {result}. Length: {len(result)}."
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
                logger.debug(
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
                        elif isinstance(datum, (Series, DataFrame)):
                            new_data.append(datum.to_dict())
                    data = new_data
                    del new_data
                    logger.info(f"Type of data: {type(data)}")
                    logger.info(f"Length of data: {len(data)}")
                elif isinstance(data, (dict, int, float, str, bool)):
                    pass
                else:  # pragma: no cover
                    raise ValueError(f"Unknown data type {type(data)} for {filename}.")
                try:
                    if isinstance(data, DataFrame):
                        data.to_json(
                            filename,
                            index=False,
                            force_ascii=False,
                            indent=4,
                        )
                    elif isinstance(data, dict):
                        Series(data).to_json(
                            filename,
                            index=False,
                            force_ascii=False,
                            indent=4,
                        )
                    elif isinstance(data, list):
                        Series(data).to_json(
                            filename,
                            index=False,
                            force_ascii=False,
                            indent=4,
                        )
                    elif isinstance(data, (int, float, str, bool)):
                        data = [data]
                        Series(data).to_json(
                            filename,
                            index=False,
                            force_ascii=False,
                            indent=4,
                        )
                    else:
                        raise ValueError(
                            f"Unknown data type {type(data)} for {filename}.",
                        )
                except ValueError as e:
                    if "using all scalar values" in str(e):
                        # Sort the dictionary by key
                        data = dict(sorted(data.items()))
                        # Save the dictionary to a JSON file
                        Series(data).to_json(
                            filename,
                            index=False,
                            force_ascii=False,
                            indent=4,
                        )
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
                raise ValueError(f"Unknown file type {suffix} for {suffix}")
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
        data = self.initialize(data_file)

        assert isinstance(data, list), f"Data is not a list: {type(data)}"
        assert len(data) == 4, f"Data is not generated: {data}. Length: {len(data)},"
        if data_file is not None:
            self.save(data=data, filename=data_file)
        if train_labels_file is not None:
            self.save(data[2], train_labels_file)
        if test_labels_file is not None:
            self.save(data[3], test_labels_file)
        return data
