import unittest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree
import os
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from deckard.layers.utils import find_conf_files, get_overrides, compose_experiment, save_params_file



this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


class testFindConfFiles(unittest.TestCase):
    config_dir = Path(this_dir, "../conf/experiment").resolve().as_posix()
    config_file = "evasion.yaml"

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.dir = mkdtemp()
        self.cfg["files"]["directory"] = self.dir
        self.exp = instantiate(config=self.cfg)

    def test_find_conf_files_from_name(self):
        files = find_conf_files(
            config_name=self.config_file,
            config_subdir="experiment",
            config_dir=Path(self.config_dir).parent,
        )
        self.assertEqual(Path(files[0]).name, self.config_file)
        self.assertEqual(Path(files[0]).parent.name, Path(self.config_dir).name)
    
    def test_find_conf_files_from_regex(self):
        files = find_conf_files(
            config_regex="*.yaml",
            config_subdir="experiment",
            config_dir=Path(self.config_dir).parent,
        )
        self.assertEqual(Path(files[0]).name, self.config_file)
        self.assertEqual(Path(files[0]).parent.name, Path(self.config_dir).name)
    
    def test_find_conf_files_from_default(self):
        files = find_conf_files(
            default_file=Path(self.config_dir, "experiment", self.config_file),
            config_subdir="experiment",
            config_dir=Path(self.config_dir).parent,
        )
        self.assertEqual(Path(files[0]).name, self.config_file)
        self.assertEqual(Path(files[0]).parent.name, Path(self.config_dir).name)
        
    def tearDown(self) -> None:
        rmtree(self.dir)

class testGetOverrides(unittest.TestCase):
    file = "evasion.yaml"
    overrides = ["++data.sample.random_state=420"]
    config_dir = Path(this_dir, "../conf/experiment").resolve().as_posix()
    
    def setup(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.file)
        self.cfg = cfg

    def test_override(self):
        overrides = get_overrides(file=self.file, folder=self.config_dir, overrides=self.overrides)
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.file, overrides=overrides)
        self.assertEqual(cfg.data.sample.random_state, 420)

class testGetOverridesFromString(testGetOverrides):
    file = "evasion.yaml"
    overrides = "++data.sample.random_state=420"
    config_dir = Path(this_dir, "../conf/experiment").resolve().as_posix()
    
class testComposeExperiment(unittest.TestCase):
    file = "evasion.yaml"
    overrides = ["data.sample.random_state=420", "data.sample.train_size=100"]
    config_dir = Path(this_dir, "../conf/experiment").resolve().as_posix()
    
    def setup(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.file)
        self.cfg = cfg

    def test_compose(self):
        exp = compose_experiment(file=self.file, config_dir=self.config_dir, overrides=self.overrides, default_file=self.file)
        self.assertEqual(exp.data.sample.random_state, 420)

class testSaveParamsFile(unittest.TestCase):
    file = "evasion.yaml"
    overrides = ["++data.sample.random_state=420"]
    config_dir = Path(this_dir, "../conf/experiment").resolve().as_posix()
    params_file = "params.yaml"
    dir = mkdtemp()
    params_file = Path(dir, params_file)
    
    def setup(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.file)
        self.cfg = cfg

    def test_save(self):
        save_params_file(config_dir=self.config_dir, config_file=self.file, overrides=self.overrides, params_file=self.params_file)
        self.assertTrue(Path(self.params_file).exists())
        
    def tearDown(self) -> None:
        rmtree(self.dir)