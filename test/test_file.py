import unittest
from pathlib import Path
from deckard.file import FileConfig
import tempfile
import shutil
import time


class TestFileConfig(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for testing
        self.temp_dirs = {}
        for d in [
            "result_directory",
            "model_directory",
            "data_directory",
            "log_directory",
        ]:
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs[d] = temp_dir

        self.config = FileConfig(
            log_file="{experiment_name}.log",
            model_file="{experiment_name}.pkl",
            data_file="{experiment_name}.csv",
            score_file="{experiment_name}_score.txt",
            replace={
                "{hash}" : "null",
                "{timestamp}" : str(time.time()),     
                "{experiment_name}" : "foo",         
            }
        )

    def tearDown(self):
        # Remove temporary directories
        for d in self.temp_dirs.values():
            shutil.rmtree(d, ignore_errors=True)
        # Remove any additional test_models2 directory if created
        if Path("test_models2").exists():
            shutil.rmtree("test_models2", ignore_errors=True)


    def test_file_paths_contain_experiment_name(self):
        exp_name = self.config.replace["{experiment_name}"]
        self.assertIn(exp_name, self.config.model_file)
        self.assertIn(exp_name, self.config.data_file)
        self.assertIn(exp_name, self.config.log_file)

    def test_file_dict(self):
        files_dict = self.config._file_dict
        self.assertIn("model_file", files_dict)
        self.assertIn("data_file", files_dict)
        self.assertIn("log_file", files_dict)
        self.assertIn("score_file", files_dict)
        self.assertTrue(files_dict["model_file"].endswith(".pkl"))
        self.assertTrue(files_dict["data_file"].endswith(".csv"))
        self.assertTrue(files_dict["log_file"].endswith(".log"))
        self.assertTrue(files_dict["score_file"].endswith("_score.txt"))

    def test_hash_placeholder(self):
        config = FileConfig(replace=self.config.replace, attack_file="{hash}")
        self.assertNotEqual(config.attack_file, "{hash}")

    def test_timestampe_placeholder(self):
        config = FileConfig(replace=self.config.replace, attack_file="{timestamp}")
        self.assertNotEqual(config.attack_file, "{timestamp}")
    
    def test_timestampe_placeholder(self):
        self.assertEqual(self.config.log_file, "foo.log")
     
    def test_unused_directory_removed(self):
        config = FileConfig()
        self.assertFalse(hasattr(config, "foo"))
