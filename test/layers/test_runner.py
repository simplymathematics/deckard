import shutil
import logging
import unittest
import os
import yaml
import numpy as np
from copy import deepcopy
from pathlib import Path
import subprocess
from deckard.base.experiment import Experiment, config

# TODO other names

logger = logging.basicConfig(level=logging.DEBUG)
class testRunner(unittest.TestCase):
    def setUp(self, config=config):
        self.here = Path(__file__).parent
        self.path = Path(self.here, "..", "..", "examples", "classification")
    
    def test_runner(self):
        os.chdir(self.path)
        cmd = "python -m deckard.layers.runner train"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, cwd = self.path, shell=True)
        for line in iter(p.stdout.readline, b''):
            print(line)
        p.stdout.close()
        exit_code = p.wait()
        self.assertEqual(exit_code, 0)
    def tearDown(self) -> None:
        from shutil import rmtree
        if Path("model").is_dir():
            rmtree("model")
        if Path("data").is_dir():
            rmtree("data")
        if Path("reports").is_dir():
            rmtree("reports")
        del self.path