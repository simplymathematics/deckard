import shutil
from pathlib import Path
from tempfile import mkdtemp

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from unittest.mock import patch

from deckard.data import DataConfig

try:
    import seaborn  # noqa: F401

    from deckard.plugins.seaborn.plot import (
        SeabornPlotConfig,
        SeabornPlotConfigList,
    )
except Exception:
    pytest.skip("seaborn is required for seaborn plot tests", allow_module_level=True)


matplotlib.use("Agg")


class TestSeabornPlots:
    @classmethod
    def setup_class(cls):
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
    def teardown_class(cls):
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

        assert isinstance(ax, Axes)
        assert out_file.exists()
        assert ax.get_title() == "My Scatter"
        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        legend = ax.get_legend()
        if legend is not None:
            assert legend.get_title().get_text() == "Group"

    def test_call_uses_passed_axes(self):
        fig, ax = plt.subplots()
        cfg = self._make_cfg()
        returned = cfg(ax=ax)
        assert returned is ax
        plt.close(fig)

    def test_heatmap_plot_uses_matrix_data(self):
        corr = pd.DataFrame(
            [[1.0, 0.25], [0.25, 1.0]],
            columns=["x", "y"],
            index=["x", "y"],
        )
        cfg = SeabornPlotConfig(
            plot_type="heatmap",
            x="x",
            y="y",
            data=corr,
            kwargs={"annot": True},
        )

        ax = cfg()

        assert isinstance(ax, Axes)

    def test_cat_plot_returns_real_axis(self):
        cfg = self._make_cfg(
            plot_type="cat",
            x="group",
            y="y",
            kwargs={"kind": "box"},
        )

        ax = cfg()

        assert isinstance(ax, Axes)
        assert len(ax.patches) + len(ax.lines) > 0

    def test_post_init_raises_for_missing_data_file(self):
        missing_file = (self.temp_dir / "does_not_exist.pkl").as_posix()
        with pytest.raises(AssertionError):
            self._make_cfg(data_file=missing_file)

    def test_post_init_raises_for_missing_columns(self):
        bad_file = self.temp_dir / "bad_data.pkl"
        pd.DataFrame({"x": [1, 2, 3]}).to_pickle(bad_file)

        with pytest.raises(AssertionError):
            self._make_cfg(data_file=bad_file.as_posix(), y="y")

    def test_seaborn_config_accepts_data_config_source(self):
        cfg_data = DataConfig(
            name="make_classification",
            data_params={
                "n_samples": 20,
                "n_features": 4,
                "n_informative": 3,
                "n_redundant": 0,
                "n_repeated": 0,
                "random_state": 1,
            },
            target=None,
        )
        cfg_data()
        cfg_data._X = pd.DataFrame(cfg_data._X)
        cfg_data._X.columns = ["x", "y", "z", "w"]

        cfg = SeabornPlotConfig(
            plot_type="scatter",
            x="x",
            y="y",
            data_config=cfg_data,
        )
        ax = cfg()
        assert isinstance(ax, Axes)

    @patch("deckard.plugins.seaborn.plot.load_optuna_studies_dataframe")
    def test_seaborn_config_accepts_optuna_storage_source(self, mock_optuna_loader):
        mock_optuna_loader.return_value = pd.DataFrame(
            {
                "trial": [1, 2, 3],
                "objective": [0.1, 0.2, 0.3],
            },
        )
        cfg = SeabornPlotConfig(
            plot_type="line",
            x="trial",
            y="objective",
            optuna_storage="sqlite:///optuna.db",
            optuna_study_name="demo",
            optuna_query={"study_names": ["demo", "other"], "row_slice": (0, 2)},
        )
        ax = cfg()
        assert isinstance(ax, Axes)
        kwargs = mock_optuna_loader.call_args.kwargs
        assert kwargs["study_names"] == ["demo", "other"]
        assert kwargs["row_slice"] == (0, 2)

    def test_plot_config_list_len(self):
        cfg_list = SeabornPlotConfigList(
            plots=["scatter", "line"],
            data_file=self.data_file.as_posix(),
        )
        assert len(cfg_list) == 2

    def test_plot_config_list_iter_current_behavior(self):
        cfg_list = SeabornPlotConfigList(
            plots=["scatter"],
            data_file=self.data_file.as_posix(),
        )
        assert next(iter(cfg_list)) == "scatter"

    def test_plot_config_list_call_current_behavior(self):
        cfg_list = SeabornPlotConfigList(
            plots=["scatter"],
            data_file=self.data_file.as_posix(),
        )
        with pytest.raises(AttributeError):
            cfg_list()
