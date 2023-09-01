import unittest
from pathlib import Path
import numpy as np
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
import os
from deckard.base.data.sampler import SklearnDataSampler

this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()
config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
config_file = "classification.yaml"


class testSklearnDataSampler(unittest.TestCase):
    def setUp(self, config_dir=config_dir, config_file=config_file):
        with initialize_config_dir(
            config_dir=Path(config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)

    def test_init(self):
        self.assertTrue(isinstance(self.data.sample, SklearnDataSampler))

    def test_call(self):
        X, y = self.data.generate()
        data = self.data.sample(X, y)
        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], np.ndarray)
        self.assertIsInstance(data[1], np.ndarray)
        self.assertEqual(data[0].shape[0], data[2].shape[0])
        self.assertEqual(data[1].shape[0], data[3].shape[0])
        self.assertEqual(len(data), 4)

    def test_hash(self):
        old_hash = hash(self.data.sample)
        self.assertIsInstance(old_hash, int)
        new_cfg = self.cfg
        new_data = instantiate(config=new_cfg)
        new_hash = hash(new_data.sample)
        self.assertEqual(old_hash, new_hash)

    def tearDown(self) -> None:
        pass


config_file = "time_series.yaml"


class testTimeSeriesSklearnDataSampler(unittest.TestCase):
    def setUp(self, config_dir=config_dir, config_file=config_file):
        with initialize_config_dir(
            config_dir=Path(config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)

    def test_init(self):
        self.assertTrue(isinstance(self.data.sample, SklearnDataSampler))

    def test_call(self):
        X, y = self.data.generate()
        data = self.data.sample(X, y)
        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], np.ndarray)
        self.assertIsInstance(data[1], np.ndarray)
        self.assertEqual(data[0].shape[0], data[2].shape[0])
        self.assertEqual(data[1].shape[0], data[3].shape[0])
        self.assertEqual(len(data), 4)

    def test_hash(self):
        old_hash = hash(self.data.sample)
        self.assertIsInstance(old_hash, int)
        new_cfg = self.cfg
        new_data = instantiate(config=new_cfg)
        new_hash = hash(new_data.sample)
        self.assertEqual(old_hash, new_hash)

    def tearDown(self) -> None:
        pass
