import unittest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree
import os
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from deckard.base.experiment import Experiment


this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


class testExperiment(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/experiment").resolve().as_posix()
    config_file = "evasion.yaml"


    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.dir = mkdtemp()
        self.cfg['files']['directory'] = self.dir
        self.exp = instantiate(config=self.cfg)

    def test_init(self):
        self.assertTrue(isinstance(self.exp, Experiment))

    def test_hash(self):
        old_hash = hash(self.exp)
        self.assertIsInstance(old_hash, int)
        new_cfg = self.cfg
        new_exp = instantiate(config=new_cfg)
        new_hash = hash(new_exp)
        self.assertEqual(old_hash, new_hash)
    
    def test_call(self):
        files = self.exp.files()
        self.assertTrue(isinstance(files, dict))
        scores = self.exp()
        self.assertTrue(isinstance(scores, dict))
        

    def tearDown(self) -> None:
        rmtree(self.dir)


class testPoisoningAttack(testExperiment):
# TODO: Fix this class
    config_dir = Path(this_dir, "../../conf/experiment").resolve().as_posix()
    config_file = "poisoning.yaml"


# class testInferenceAttack(testExperiment):
# TODO: Fix this class. Output not compatible for sklearn.metrics.accuracy_score
#     config_dir = Path(this_dir, "../../conf/experiment").resolve().as_posix()
#     config_file = "inference.yaml"


class testExtractionAttack(testExperiment):
    config_dir = Path(this_dir, "../../conf/experiment").resolve().as_posix()
    config_file = "extraction.yaml"
