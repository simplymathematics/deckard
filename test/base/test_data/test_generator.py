import unittest
from pathlib import Path
import os
import numpy as np
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from tempfile import mkdtemp
from shutil import rmtree

from deckard.base.data.generator import (
    DataGenerator,
    SklearnDataGenerator,
    TorchDataGenerator,
    KerasDataGenerator,
)


this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()
config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
config_file = "classification.yaml"


class testDataGenerator(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "classification.yaml"
    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)

    def test_init(self):
        self.assertTrue(isinstance(self.data.generate, DataGenerator))

    def test_call(self):
        data = self.data.generate()
        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], np.ndarray)
        self.assertIsInstance(data[1], np.ndarray)
        self.assertEqual(data[0].shape[0], data[1].shape[0])
        self.assertEqual(len(data), 2)

    def test_hash(self):
        old_hash = hash(self.data.generate)
        self.assertIsInstance(old_hash, int)
        new_cfg = self.cfg
        new_data = instantiate(config=new_cfg)
        new_hash = hash(new_data.generate)
        self.assertEqual(old_hash, new_hash)

    def tearDown(self) -> None:
        pass


class testDataGeneratorSklearn(testDataGenerator):
    config_file = "regression.yaml"

class testDataGeneratorTorch(testDataGenerator):
    config_file = "torch_mnist.yaml"
    
class testDataGeneratorKeras(testDataGenerator):
    config_file = "keras_mnist.yaml"
    
class testDataGeneratorTensorflow(testDataGenerator):
    config_file = "tensorflow_mnist.yaml"

class testSklearnDataGenerator(unittest.TestCase):
    def setUp(self):
        self.names = ["classification", "regression", "blobs", "moons", "circles"]

    def test_init(self):
        for name in self.names:
            data = SklearnDataGenerator(name=name)
            self.assertTrue(isinstance(data, SklearnDataGenerator))

    def test_call(self):
        for name in self.names:
            data = SklearnDataGenerator(name=name)()
            self.assertIsInstance(data, list)
            self.assertIsInstance(data[0], np.ndarray)
            self.assertIsInstance(data[1], np.ndarray)
            self.assertEqual(data[0].shape[0], data[1].shape[0])
            self.assertEqual(len(data), 2)

    def test_hash(self):
        for name in self.names:
            data = SklearnDataGenerator(name=name)
            old_hash = hash(data)
            self.assertIsInstance(old_hash, int)
            new_data = SklearnDataGenerator(name=name)
            new_hash = hash(new_data)
            self.assertEqual(old_hash, new_hash)

    def tearDown(self) -> None:
        pass


class testTorchDataGenerator(unittest.TestCase):
    def setUp(self):
        self.names = ["torch_mnist", "torch_cifar", "torch_diabetes", "torch_cifar100"]
        self.dir = Path(Path.cwd(), "tmp" ).resolve().as_posix()
        Path(self.dir).mkdir(parents=True, exist_ok=True)

    def test_init(self):
        for name in self.names:
            data = TorchDataGenerator(name=name, path=self.dir)
            self.assertTrue(isinstance(data, TorchDataGenerator))

    def test_hash(self):
        for name in self.names:
            data = TorchDataGenerator(name=name, path=self.dir)
            old_hash = hash(data)
            self.assertIsInstance(old_hash, int)
            new_data = TorchDataGenerator(name=name, path=self.dir)
            new_hash = hash(new_data)
            self.assertEqual(old_hash, new_hash)

    def test_call(self):
        for name in self.names:
            data = TorchDataGenerator(name=name, path=self.dir)()
            data = TorchDataGenerator(name=name, path=self.dir)() # Test it again to make sure download only happens once
            self.assertIsInstance(data, list)
            self.assertIsInstance(data[0], np.ndarray)
            self.assertIsInstance(data[1], np.ndarray)
            self.assertEqual(data[0].shape[0], data[1].shape[0])
            self.assertEqual(len(data), 2)

    def tearDown(self) -> None:
        rmtree(self.dir)
        

class testKerasDataGenerator(unittest.TestCase):
        
    def setUp(self):
        self.names = ["keras_mnist", "keras_cifar10", "mnist", "cifar10", "diabetes"]
        self.dir = Path(Path.cwd(), "tmp" ).resolve().as_posix()
        Path(self.dir).mkdir(parents=True, exist_ok=True)
    
    def test_init(self):
        for name in self.names:
            data = KerasDataGenerator(name=name, path=self.dir)
            self.assertTrue(isinstance(data, KerasDataGenerator))

    def test_hash(self):
        for name in self.names:
            data = KerasDataGenerator(name=name, path=self.dir)
            old_hash = hash(data)
            self.assertIsInstance(old_hash, int)
            new_data = KerasDataGenerator(name=name, path=self.dir)
            new_hash = hash(new_data)
            self.assertEqual(old_hash, new_hash)

    def test_call(self):
        for name in self.names:
            data = KerasDataGenerator(name=name, path=self.dir)()
            self.assertIsInstance(data, list)
            self.assertIsInstance(data[0], np.ndarray)
            self.assertIsInstance(data[1], np.ndarray)
            self.assertEqual(data[0].shape[0], data[1].shape[0])
            self.assertEqual(len(data), 2)

    def tearDown(self) -> None:
        rmtree(self.dir)
        