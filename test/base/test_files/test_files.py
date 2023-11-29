import unittest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree
import os
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from deckard.base.files import FileConfig

this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


class testFiles(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/files").resolve().as_posix()
    config_file = "default.yaml"

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.directory = mkdtemp()
        self.cfg["directory"] = self.directory
        self.files = instantiate(config=self.cfg)

    def test_init(self):
        self.assertIsInstance(self.files, FileConfig)

    def test_hash(self):
        old_hash = hash(self.files)
        self.assertIsInstance(old_hash, int)
        new_hash = hash(instantiate(config=self.cfg))
        self.assertEqual(old_hash, new_hash)
        self.files()
        after_call = hash(self.files)
        self.assertEqual(old_hash, after_call)

    def test_check_status(self):
        files = self.files()
        file = list(files.values())[0]
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        Path(file).touch()
        status = self.files.check_status()
        self.assertEqual(list(status.values())[0], True)

    def test_get_filenames(self):
        files = self.files()
        filenames = self.files.get_filenames()
        self.assertEqual(len(files), len(filenames))

    def tearDown(self) -> None:
        rmtree(self.directory)


class SklearnTestFiles(testFiles):
    config_dir = Path(this_dir, "../../conf/files").resolve().as_posix()
    config_file = "sklearn.yaml"


class TorchTestFiles(testFiles):
    config_dir = Path(this_dir, "../../conf/files").resolve().as_posix()
    config_file = "torch.yaml"


class KerasTestFiles(testFiles):
    config_dir = Path(this_dir, "../../conf/files").resolve().as_posix()
    config_file = "keras.yaml"


class TensorflowTestFiles(testFiles):
    config_dir = Path(this_dir, "../../conf/files").resolve().as_posix()
    config_file = "tensorflow.yaml"


# class TensorflowTestFiles(testFiles):
#     config_dir = Path(this_dir, "../../conf/files").resolve().as_posix()
#     config_file = "sklearn_no_model.yaml"
