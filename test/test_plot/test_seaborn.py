import unittest
from pathlib import Path
from tempfile import mkdtemp
import shutil
import pytest
import matplotlib
from matplotlib.axes import Axes
import pandas as pd

import matplotlib.pyplot as plt

pytest.importorskip("seaborn")

from deckard.plot.seaborn_plots import (  # NOQA E402
    SeabornPlotConfig,
    SeabornPlotConfigList,
)  # NOQA E402


matplotlib.use("Agg")


class TestSeabornPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(mkdtemp())
        cls.data_file = cls.temp_dir / "seaborn_data.pkl"
        cls.df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [2, 4, 6, 8],
                "group": ["a", "a", "b", "b"],
                "style_col": ["s1", "s2", "s1", "s2"],
            },
        )
        cls.df.to_pickle(cls.data_file)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _make_cfg(self, **overrides):
        params = dict(
            plot_type="scatter",
            x="x",
            y="y",
            kwargs={},
            rc_config={},
            data_file=self.data_file.as_posix(),
            title=None,
            xlabel=None,
            ylabel=None,
            xscale=None,
            yscale=None,
            hue=None,
            style=None,
            plot_file=None,
            legend_title=None,
        )
        params.update(overrides)
        return SeabornPlotConfig(**params)

    def test_scatter_plot_saves_and_sets_labels(self):
        out_file = self.temp_dir / "scatter.png"
        cfg = self._make_cfg(
            plot_file=out_file.as_posix(),
            title="My Scatter",
            xlabel="X Label",
            ylabel="Y Label",
            xscale="linear",
            yscale="linear",
            hue="group",
            legend_title="Group",
        )
        ax = cfg()

        self.assertIsInstance(ax, Axes)
        self.assertTrue(out_file.exists())
        self.assertEqual(ax.get_title(), "My Scatter")
        self.assertEqual(ax.get_xlabel(), "X Label")
        self.assertEqual(ax.get_ylabel(), "Y Label")
        legend = ax.get_legend()
        if legend is not None:
            self.assertEqual(legend.get_title().get_text(), "Group")

    def test_call_uses_passed_axes(self):
        fig, ax = plt.subplots()
        cfg = self._make_cfg()
        returned = cfg(ax=ax)
        self.assertIs(returned, ax)
        plt.close(fig)

    def test_post_init_raises_for_missing_data_file(self):
        missing_file = (self.temp_dir / "does_not_exist.pkl").as_posix()
        with self.assertRaises(AssertionError):
            self._make_cfg(data_file=missing_file)

    def test_post_init_raises_for_missing_columns(self):
        bad_file = self.temp_dir / "bad_data.pkl"
        pd.DataFrame({"x": [1, 2, 3]}).to_pickle(bad_file)

        with self.assertRaises(AssertionError):
            self._make_cfg(data_file=bad_file.as_posix(), y="y")

    def test_plot_config_list_len(self):
        cfg_list = SeabornPlotConfigList(
            plots=["scatter", "line"],
            data_file=self.data_file.as_posix(),
        )
        self.assertEqual(len(cfg_list), 2)

    def test_plot_config_list_iter_current_behavior(self):
        cfg_list = SeabornPlotConfigList(
            plots=["scatter"],
            data_file=self.data_file.as_posix(),
        )
        self.assertEqual(next(iter(cfg_list)), "scatter")

    def test_plot_config_list_call_current_behavior(self):
        cfg_list = SeabornPlotConfigList(
            plots=["scatter"],
            data_file=self.data_file.as_posix(),
        )
        with self.assertRaises(AttributeError):
            cfg_list()


if __name__ == "__main__":
    unittest.main()
