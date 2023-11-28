import unittest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree
import os
import numpy as np
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from pandas import DataFrame, Series

from deckard.base.data import Data

this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


class testSklearnData(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "titanic.yaml"
    data_type = ".json"
    data_file = "data"

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(), version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)
        self.directory = mkdtemp()

    def test_init(self):
        self.assertTrue(isinstance(self.data, Data))

    def test_call(self):
        filename = Path(self.directory, self.data_file + self.data_type).as_posix()
        X_train, X_test, y_train, y_test = self.data(data_file=filename)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])
        self.assertTrue(Path(filename).exists())

    def test_hash(self):
        old_hash = hash(self.data)
        self.assertIsInstance(old_hash, int)
        new_cfg = self.cfg
        new_data = instantiate(config=new_cfg)
        new_hash = hash(new_data)
        self.assertEqual(old_hash, new_hash)
        new_data()
        hash_after_call = hash(new_data)
        self.assertEqual(old_hash, hash_after_call)

    def test_initialize(self):
        data = self.data.initialize()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 4)

    def test_load(self):
        data_file = Path(self.directory, self.data_file + self.data_type).as_posix()
        _ = self.data(data_file=data_file)
        data = self.data.load(data_file)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 4)

    def test_save(self):
        data_file = Path(self.directory, self.data_file + self.data_type).as_posix()
        data = self.data(data_file)
        self.assertTrue(Path(data_file).exists())
        self.data.save(data, data_file)
        self.assertTrue(Path(data_file).exists())

    def test_resave(self):
        data_file = Path(self.directory, self.data_file + self.data_type).as_posix()
        train_labels_file = Path(data_file).with_suffix(".csv").as_posix()
        test_labels_file = Path(data_file).with_suffix(".json").as_posix()
        _ = self.data(
            train_labels_file=train_labels_file, test_labels_file=test_labels_file,
        )
        score_dict = {"test_score": 0.5}
        score_series = Series(score_dict)
        score_df = DataFrame(score_dict, index=[0])
        self.data.save(score_dict, data_file)
        self.assertTrue(Path(data_file).exists())
        self.data.save(score_series, data_file)
        self.assertTrue(Path(data_file).exists())
        self.data.save(score_df, data_file)
        self.assertTrue(Path(data_file).exists())

    def test_load_from_disk(self):
        data_file = Path(self.directory, self.data_file + self.data_type).as_posix()
        data = self.data(data_file)
        self.assertTrue(Path(data_file).exists())
        data = self.data(data_file=data_file)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 4)
        data = self.data.initialize(filename=data_file)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 4)

    def tearDown(self) -> None:
        rmtree(self.directory)


class testKerasData(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "keras_mnist.yaml"
    data_type = ".pkl"
    data_file = "data"


class testTorchData(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "torch_mnist.yaml"
    data_type = ".pkl"
    data_file = "data"


class testTensorflowData(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "tensorflow_mnist.yaml"
    data_type = ".pkl"
    data_file = "data"
