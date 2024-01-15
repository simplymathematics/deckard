import unittest
from pathlib import Path
from tempfile import mkdtemp
import os
import numpy as np
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from deckard.base.data.sklearn_pipeline import (
    SklearnDataPipelineStage,
    SklearnDataPipeline,
)


this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()
config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
config_file = "classification.yaml"


class testSklearnDataPipeline(unittest.TestCase):
    def setUp(self, config_dir=config_dir, config_file=config_file):
        with initialize_config_dir(
            config_dir=Path(config_dir).resolve().as_posix(), version_base="1.3",
        ):
            cfg = compose(config_name=config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)
        self.dir = mkdtemp()
        self.data_file = "test"
        self.data_type = "pkl"

    def test_init(self):
        self.assertTrue(isinstance(self.data.sklearn_pipeline, SklearnDataPipeline))

    def test_call(self):
        X, y = self.data.generate()
        X_train, X_test, y_train, y_test = self.data.sample(X, y)
        old_mean = np.mean(X_train)
        X_train, X_test, y_train, y_test = self.data.sklearn_pipeline(
            X_train, X_test, y_train, y_test,
        )
        new_mean = np.mean(X_train)
        self.assertNotEqual(old_mean, new_mean)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])

    def test_hash(self):
        old_hash = hash(self.data.sklearn_pipeline)
        self.assertIsInstance(old_hash, int)
        new_cfg = self.cfg
        new_data = instantiate(config=new_cfg)
        new_hash = hash(new_data.sklearn_pipeline)
        self.assertEqual(old_hash, new_hash)
        X, y = new_data.generate()
        data = new_data.sample(X, y)
        data = new_data.sklearn_pipeline(*data)
        hash_after_call = hash(new_data.sklearn_pipeline)
        self.assertEqual(old_hash, hash_after_call)

    def test_len(self):
        self.assertIsInstance(len(self.data.sklearn_pipeline), int)

    def test_getitem(self):
        for stage in self.data.sklearn_pipeline:
            self.assertIsInstance(
                self.data.sklearn_pipeline[stage], SklearnDataPipelineStage,
            )

    def tearDown(self) -> None:
        pass


class testSklearnDataPipelineStage(unittest.TestCase):
    def setUp(self, config_dir=config_dir, config_file=config_file):
        with initialize_config_dir(
            config_dir=Path(config_dir).resolve().as_posix(), version_base="1.3",
        ):
            cfg = compose(config_name=config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)

    def test_init(self):
        self.assertTrue(
            isinstance(
                self.data.sklearn_pipeline["preprocessor"], SklearnDataPipelineStage,
            )
            or hasattr(self.data.sklearn_pipeline["preprocessor"], "transform"),
        )

    def test_call(self):
        X, y = self.data.generate()
        X_train, X_test, y_train, y_test = self.data.sample(X, y)
        old_mean = np.mean(X_train)
        X_train, X_test, y_train, y_test = self.data.sklearn_pipeline["preprocessor"](
            X_train, X_test, y_train, y_test,
        )
        new_mean = np.mean(X_train)
        self.assertNotEqual(old_mean, new_mean)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])

    def test_hash(self):
        old_hash = hash(self.data.sklearn_pipeline["preprocessor"])
        self.assertIsInstance(old_hash, int)
        new_cfg = self.cfg
        new_data = instantiate(config=new_cfg)
        new_hash = hash(new_data.sklearn_pipeline["preprocessor"])
        self.assertEqual(old_hash, new_hash)
        X, y = new_data.generate()
        data = new_data.sample(X, y)
        data = new_data.sklearn_pipeline(*data)
        hash_after_call = hash(new_data.sklearn_pipeline["preprocessor"])
        self.assertEqual(old_hash, hash_after_call)

    def tearDown(self) -> None:
        pass
