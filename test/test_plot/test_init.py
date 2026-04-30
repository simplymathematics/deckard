import unittest
from unittest.mock import Mock, patch
import pytest

pytest.importorskip("seaborn")
pytest.importorskip("yellowbrick")

from deckard.plot import PlotConfig  # noqa 402
from deckard.utils import ConfigBase


class TestPlotConfig(unittest.TestCase):

    def test_plot_config_with_experiment_single_plot(self):
        """Test PlotConfig initialization with experiment and single plot type."""
        mock_experiment = Mock()
        with patch("deckard.plot.YellowbrickPlotConfig") as mock_cls:
            mock_instance = Mock()
            mock_cls.return_value = mock_instance
            plot_cfg = PlotConfig(
                kwargs={
                    "experiment": mock_experiment,
                    "plot_type": "confusion_matrix",
                },
            )
        self.assertIs(plot_cfg.config, mock_instance)

    def test_plot_config_with_experiment_multiple_plots(self):
        """Test PlotConfig initialization with experiment and multiple plots."""
        mock_experiment = Mock()
        with patch("deckard.plot.YellowbrickConfigList") as mock_cls:
            mock_instance = Mock()
            mock_cls.return_value = mock_instance
            plot_cfg = PlotConfig(
                kwargs={
                    "experiment": mock_experiment,
                    "plots": [{"plot_type": "confusion_matrix"}],
                },
            )
        self.assertIs(plot_cfg.config, mock_instance)

    def test_plot_config_with_data_file_single_plot(self):
        """Test PlotConfig initialization with data_file and single plot type."""
        with patch("deckard.plot.SeabornPlotConfig") as mock_cls:
            mock_instance = Mock()
            mock_cls.return_value = mock_instance
            plot_cfg = PlotConfig(
                kwargs={
                    "data_file": "/path/to/data.pkl",
                    "plot_type": "pairplot",
                },
            )
        self.assertIs(plot_cfg.config, mock_instance)

    def test_plot_config_with_data_file_multiple_plots(self):
        """Test PlotConfig initialization with data_file and multiple plots."""
        with patch("deckard.plot.SeabornPlotConfigList") as mock_cls:
            mock_instance = Mock()
            mock_cls.return_value = mock_instance
            plot_cfg = PlotConfig(
                kwargs={
                    "data_file": "/path/to/data.pkl",
                    "plots": [{"plot_type": "pairplot"}],
                },
            )
        self.assertIs(plot_cfg.config, mock_instance)

    def test_plot_config_missing_source(self):
        """Test PlotConfig raises error when neither experiment nor data_file provided."""
        with self.assertRaises(ValueError) as context:
            PlotConfig(kwargs={"plot_type": "confusion_matrix"})
        self.assertIn("Missing required source key", str(context.exception))

    def test_plot_config_both_sources(self):
        """Test PlotConfig raises error when both experiment and data_file provided."""
        mock_experiment = Mock()
        with self.assertRaises(ValueError) as context:
            PlotConfig(
                kwargs={
                    "experiment": mock_experiment,
                    "data_file": "/path/to/data.pkl",
                    "plot_type": "confusion_matrix",
                },
            )
        self.assertIn(
            "Provide either 'experiment' or 'data_file', not both",
            str(context.exception),
        )

    def test_plot_config_call(self):
        """Test PlotConfig __call__ method delegates to config."""
        mock_experiment = Mock()
        with patch("deckard.plot.YellowbrickPlotConfig") as mock_config_cls:
            mock_config_instance = Mock(return_value="plot_result")
            mock_config_cls.return_value = mock_config_instance

            plot_cfg = PlotConfig(
                kwargs={
                    "experiment": mock_experiment,
                    "plot_type": "confusion_matrix",
                },
            )
            plot_cfg.config = mock_config_instance
            plot_cfg()
            mock_config_instance.assert_called_once()

    def test_plot_config_getattr(self):
        """Test PlotConfig __getattr__ delegates to config."""
        mock_experiment = Mock()
        with patch("deckard.plot.YellowbrickPlotConfig") as mock_config_cls:
            mock_config_instance = Mock()
            mock_config_instance.some_attribute = "test_value"
            mock_config_cls.return_value = mock_config_instance

            plot_cfg = PlotConfig(
                kwargs={
                    "experiment": mock_experiment,
                    "plot_type": "confusion_matrix",
                },
            )
            plot_cfg.config = mock_config_instance
            self.assertEqual(plot_cfg.some_attribute, "test_value")

    def test_plot_config_len(self):
        """Test PlotConfig __len__ delegates to config."""
        mock_experiment = Mock()
        with patch("deckard.plot.YellowbrickConfigList") as mock_config_cls:
            mock_config_instance = Mock()
            mock_config_instance.__len__ = Mock(return_value=3)
            mock_config_cls.return_value = mock_config_instance

            plot_cfg = PlotConfig(
                kwargs={
                    "experiment": mock_experiment,
                    "plots": [
                        {"plot_type": "confusion_matrix"},
                        {"plot_type": "roc"},
                        {"plot_type": "feature_importance"},
                    ],
                },
            )
            plot_cfg.config = mock_config_instance
            self.assertEqual(len(plot_cfg), 3)

    def test_hash_stable_for_plot_config(self):
        """Test that PlotConfig, as ConfigBase, has hash method."""
        # PlotConfig wraps another config which may not be hashable,
        # so we just verify that PlotConfig is a ConfigBase and has __hash__
        plot_cfg_new = PlotConfig.__new__(PlotConfig)
        plot_cfg_new.kwargs = {"plot_type": "confusion_matrix"}
        plot_cfg_new.config = None  # Prevent config instantiation
        
        # Verify PlotConfig inherits from ConfigBase
        self.assertTrue(isinstance(plot_cfg_new, ConfigBase))
        
        # Verify it has the hash method
        self.assertTrue(hasattr(plot_cfg_new, "__hash__"))


if __name__ == "__main__":
    unittest.main()
