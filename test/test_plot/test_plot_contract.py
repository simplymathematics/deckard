from unittest.mock import Mock, patch

import pytest

from deckard.plot import PlotConfig
from deckard.plot.canon import normalize_plot_backend


def test_normalize_plot_backend_aliases():
    assert normalize_plot_backend(None) == "seaborn"
    assert normalize_plot_backend("sns") == "seaborn"
    assert normalize_plot_backend("yellow") == "yellowbrick"
    with pytest.raises(KeyError, match="Unsupported plot backend"):
        normalize_plot_backend("matplotlib")


def test_plot_config_tracks_runtime_state_for_seaborn_dispatch():
    with patch("deckard.plot.SeabornPlotConfig") as mock_seaborn:
        mock_instance = Mock(return_value="ok")
        mock_seaborn.return_value = mock_instance

        cfg = PlotConfig(
            kwargs={
                "data_file": "/tmp/data.pkl",
                "plot_type": "scatter",
                "plot_file": "/tmp/out.png",
            },
        )
        assert cfg.plot_state["backend"] == "seaborn"
        assert cfg.files["plot_file"] == "/tmp/out.png"

        result = cfg()
        assert result == "ok"
        assert "plot_call_time" in cfg.times
        assert cfg.times["plot_call_time"] >= 0.0
        assert cfg.plot_state["rendered"] is True


def test_plot_config_rejects_mismatched_backend_and_source():
    with pytest.raises(ValueError, match="requires yellowbrick backend"):
        PlotConfig(
            kwargs={
                "experiment": Mock(),
                "plot_type": "roc_auc",
                "backend": "seaborn",
            },
        )


def test_plot_config_accepts_plot_backend_alias():
    with patch("deckard.plot.SeabornPlotConfig") as mock_seaborn:
        mock_instance = Mock(return_value="ok")
        mock_seaborn.return_value = mock_instance

        cfg = PlotConfig(
            kwargs={
                "data_file": "/tmp/data.pkl",
                "plot_type": "scatter",
                "plot_backend": "sns",
            },
        )
        assert cfg.plot_state["backend"] == "seaborn"
        assert cfg.kwargs["backend"] == "seaborn"
        assert cfg.kwargs["plot_backend"] == "seaborn"


def test_plot_config_rejects_conflicting_backend_aliases():
    with pytest.raises(ValueError, match="different backends"):
        PlotConfig(
            kwargs={
                "data_file": "/tmp/data.pkl",
                "plot_type": "scatter",
                "backend": "seaborn",
                "plot_backend": "yellowbrick",
            },
        )
