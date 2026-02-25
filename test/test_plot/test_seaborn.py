import unittest
import pandas as pd
from matplotlib.axes import Axes
from deckard.plot.seaborn_plots import SeabornPlotConfig, SeabornCatPlotConfig, SeabornPlotListConfig

class TestSeabornPlotConfigs(unittest.TestCase):

    def setUp(self):
        self.sample_dataframe = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "hue": ["A", "B", "A", "B", "A"]
        })

    def test_seaborn_plot_config_lineplot(self):
        config = SeabornPlotConfig(
            x="x",
            y="y",
            plot_type="line",
            title="Line Plot",
            xlabel="X-Axis",
            ylabel="Y-Axis"
        )
        ax = config(self.sample_dataframe)
        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.get_title(), "Line Plot")
        self.assertEqual(ax.get_xlabel(), "X-Axis")
        self.assertEqual(ax.get_ylabel(), "Y-Axis")

    def test_seaborn_plot_config_scatterplot(self):
        config = SeabornPlotConfig(
            x="x",
            y="y",
            plot_type="scatter",
            title="Scatter Plot",
            xlabel="X-Axis",
            ylabel="Y-Axis",
            hue="hue"
        )
        ax = config(self.sample_dataframe)
        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.get_title(), "Scatter Plot")
        self.assertEqual(ax.get_xlabel(), "X-Axis")
        self.assertEqual(ax.get_ylabel(), "Y-Axis")

    def test_seaborn_catplot_config(self):
        config = SeabornCatPlotConfig(
            x="x",
            y="y",
            plot_type="cat",
            title="Cat Plot",
            xlabel="X-Axis",
            ylabel="Y-Axis",
            hue="hue"
        )
        g = config(self.sample_dataframe)
        self.assertIsNotNone(g.fig)
        self.assertEqual(g.fig._suptitle.get_text(), "Cat Plot")

    def test_seaborn_plot_list_config(self):
        plot_configs = [
            SeabornPlotConfig(
                x="x",
                y="y",
                plot_type="line",
                title="Line Plot"
            ),
            SeabornPlotConfig(
                x="x",
                y="y",
                plot_type="scatter",
                title="Scatter Plot"
            )
        ]
        list_config = SeabornPlotListConfig(plot_configs=plot_configs)
        # Should execute without errors
        list_config(self.sample_dataframe)

if __name__ == "__main__":
    unittest.main()