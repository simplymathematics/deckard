import unittest
from pathlib import Path
from deckard.file import FileConfig
import tempfile
import shutil


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
            experiment_name="{timestamp}",
            result_directory=self.temp_dirs["result_directory"],
            model_directory=self.temp_dirs["model_directory"],
            data_directory=self.temp_dirs["data_directory"],
            log_directory=self.temp_dirs["log_directory"],
            log_file="{experiment_name}.log",
            model_file="{experiment_name}.pkl",
            data_file="{experiment_name}.csv",
            score_file="{experiment_name}_score.txt",
        )

    def tearDown(self):
        # Remove temporary directories
        for d in self.temp_dirs.values():
            shutil.rmtree(d, ignore_errors=True)
        # Remove any additional test_models2 directory if created
        if Path("test_models2").exists():
            shutil.rmtree("test_models2", ignore_errors=True)

    def test_directories_created(self):
        self.assertTrue(Path(self.config.result_directory).exists())
        self.assertTrue(Path(self.config.model_directory).exists())
        self.assertTrue(Path(self.config.data_directory).exists())
        self.assertTrue(Path(self.config.log_directory).exists())

    def test_experiment_name_timestamp(self):
        self.assertNotEqual(self.config.experiment_name, "{timestamp}")
        self.assertTrue(
            self.config.experiment_name.isdigit() or "-" in self.config.experiment_name,
        )

    def test_file_paths_contain_experiment_name(self):
        exp_name = self.config.experiment_name
        self.assertIn(exp_name, self.config.model_file)
        self.assertIn(exp_name, self.config.data_file)
        self.assertIn(exp_name, self.config.log_file)

    def test_call(self):
        files_dict = self.config()
        self.assertIn("model_file", files_dict)
        self.assertIn("data_file", files_dict)
        self.assertIn("log_file", files_dict)
        self.assertIn("score_file", files_dict)
        self.assertTrue(files_dict["model_file"].endswith(".pkl"))
        self.assertTrue(files_dict["data_file"].endswith(".csv"))
        self.assertTrue(files_dict["log_file"].endswith(".log"))
        self.assertTrue(files_dict["score_file"].endswith("_score.txt"))

    def test_hash_placeholder(self):
        config = FileConfig(experiment_name="{hash}")
        self.assertNotEqual(config.experiment_name, "{hash}")
        self.assertEqual(len(config.experiment_name), 32)  # md5 hash length

    def test_placeholder_replacement_in_experiment_name(self):
        config = FileConfig(experiment_name="exp_{timestamp}")
        self.assertIn("exp_", config.experiment_name)
        self.assertNotIn("{timestamp}", config.experiment_name)

    def test_unused_directory_removed(self):
        config = FileConfig()
        self.assertFalse(hasattr(config, "attack_file_directory"))
        self.assertFalse(hasattr(config, "attack_training_predictions_file_directory"))
        self.assertFalse(hasattr(config, "attack_test_predictions_file_directory"))
        self.assertFalse(hasattr(config, "score_file_directory"))
