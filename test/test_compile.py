import unittest
import pandas as pd
import json
import yaml
from pathlib import Path
from deckard.compile import ResultFolderConfig, ParetoConfig, ParetoConfigList, ResultFormatterConfig, CompileConfig

class TestDeckardCompile(unittest.TestCase):

    def setUp(self):
        self.tmp_path = Path("tmp_test_dir")
        self.tmp_path.mkdir(exist_ok=True)

    def tearDown(self):
        for item in self.tmp_path.iterdir():
            if item.is_file():
                item.unlink()
        self.tmp_path.rmdir()

    def mock_result_folder_config(self):
        # Create mock files
        params_file = self.tmp_path / "params.yaml"
        scores_file = self.tmp_path / "scores.json"
        params_file.write_text(yaml.dump({"param1": "value1", "param2": "value2"}))
        scores_file.write_text(json.dumps({"score1": 0.9, "score2": 0.8}))
        return ResultFolderConfig(directory=self.tmp_path.as_posix())

    def test_result_folder_config(self):
        config = self.mock_result_folder_config()
        self.assertEqual(len(config.params_files), 1)
        self.assertEqual(len(config.score_files), 1)
        combined_df = config._read_pair(config.score_files[0], config.params_files[0])
        self.assertFalse(combined_df.empty)
        self.assertIn("param1", combined_df.columns)
        self.assertIn("score1", combined_df.columns)

    def test_pareto_config(self):
        df = pd.DataFrame({"metric": [1, 2, 3, 4, 5]})
        config = ParetoConfig(metric="metric", direction="maximize")
        pareto_df = config(df)
        self.assertFalse(pareto_df.empty)
        self.assertEqual(pareto_df["metric"].iloc[0], 5)

    def test_pareto_config_list(self):
        df = pd.DataFrame({"metric1": [1, 2, 3], "metric2": [3, 2, 1]})
        config1 = ParetoConfig(metric="metric1", direction="maximize")
        config2 = ParetoConfig(metric="metric2", direction="minimize")
        pareto_list = ParetoConfigList(pareto_configs=[config1, config2])
        pareto_df = pareto_list(df)
        self.assertFalse(pareto_df.empty)

    def test_result_formatter_config(self):
        df = pd.DataFrame({"metric1": [1.12345, 2.6789], "metric2": [3.98765, 4.54321]})
        config = ResultFormatterConfig(metrics=["metric1", "metric2"], sig_figs=[2, 3])
        formatted_df = config(df)
        self.assertFalse(formatted_df.empty)
        self.assertAlmostEqual(formatted_df["metric1"].iloc[0], 1.12, places=2)
        self.assertAlmostEqual(formatted_df["metric2"].iloc[0], 3.988, places=3)

    def test_compile_config(self):
        compile_config_file = self.tmp_path / "compile_config.yaml"
        compile_config_file.write_text(yaml.dump({
            "directory": self.tmp_path.as_posix(),
            "params_regex": "**/params.yaml",
            "scores_regex": "**/scores.json",
            "output_file": str(self.tmp_path / "results.csv")
        }))
        # Generate mock result files
        params_file = self.tmp_path / "params.yaml"
        scores_file = self.tmp_path / "scores.json"
        params_file.write_text(yaml.dump({"param1": "value1", "param2": "value2"}))
        scores_file.write_text(json.dumps({"score1": 0.9, "score2": 0.8}))
        config = CompileConfig(output_file=self.tmp_path / "results.csv", compile_config_file=compile_config_file.as_posix())
        results = config()
        self.assertTrue(Path(config.output_file).is_file())
        self.assertFalse(results.empty)

if __name__ == "__main__":
    unittest.main()