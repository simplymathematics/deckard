import shutil
import os
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

import matplotlib
import pytest
from omegaconf import OmegaConf
from sklearn.model_selection import KFold, StratifiedKFold

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)

try:
    import yellowbrick  # noqa: F401

    from deckard.data import DataConfig
    from deckard.experiment import ExperimentConfig
    from deckard.file import FileConfig
    from deckard.model import ModelConfig
    from deckard.plugins.yellowbrick.plot import (
        YellowbrickConfigList,
        YellowbrickPlotConfig,
        cluster_viz_types,
    )
except ImportError:
    pytest.skip(
        "yellowbrick is required for yellowbrick plot tests",
        allow_module_level=True,
    )


class TestYellowbrickPlots:

    def _make_classification_experiment(self):
        files = FileConfig(data_file="", model_file="")
        data_conf = OmegaConf.load(self.classification_data_config)
        model_conf = OmegaConf.load(self.classification_model_config)
        data = DataConfig(**OmegaConf.to_container(data_conf, resolve=True))
        model = ModelConfig(**OmegaConf.to_container(model_conf, resolve=True))
        return ExperimentConfig(data=data, model=model, files=files)

    def _make_regression_experiment(self):
        files = FileConfig(data_file="", model_file="")
        data_conf = OmegaConf.load(self.regression_data_config)
        model_conf = OmegaConf.load(self.regression_model_config)
        data = DataConfig(**OmegaConf.to_container(data_conf, resolve=True))
        model = ModelConfig(**OmegaConf.to_container(model_conf, resolve=True))
        return ExperimentConfig(data=data, model=model, files=files, classifier=False)

    def _make_classification_experiment_with_persisted_files(self, stem: str):
        classification_data = OmegaConf.load(self.classification_data_config)
        classification_model = OmegaConf.load(self.classification_model_config)
        files = FileConfig(
            data_file=f"{self.temp_dir}/data/{stem}.pkl",
            model_file=f"{self.temp_dir}/models/{stem}.pkl",
        )
        return ExperimentConfig(
            data=classification_data,
            model=classification_model,
            files=files,
        )

    def _make_clustering_experiment(self, *, persisted_files: bool = False):
        if persisted_files:
            files = FileConfig(
                data_file=f"{self.temp_dir}/data/cluster_data.pkl",
                model_file=f"{self.temp_dir}/models/kmeans_model.pkl",
            )
        else:
            files = FileConfig(data_file="", model_file="")

        cluster_data = OmegaConf.load(self.cluster_data_config)
        cluster_model = OmegaConf.load(self.cluster_model_config)
        cluster_model["scorer"] = None
        return ExperimentConfig(data=cluster_data, model=cluster_model, files=files)

    def test_parse_cv_defaults_to_stratifiedkfold_for_classifier(self):
        plot_cfg = YellowbrickPlotConfig(
            experiment=self._make_classification_experiment(),
            plot_type="learning_curve",
            plot_params={},
        )

        cv = plot_cfg.parse_cv()

        assert isinstance(cv, StratifiedKFold)
        assert cv.n_splits == 5

    def test_yellowbrick_requires_experiment_config(self):
        with pytest.raises(TypeError):
            YellowbrickPlotConfig(
                experiment=object(),
                plot_type="roc_auc",
            )

    def test_parse_cv_defaults_to_kfold_for_regressor(self):
        plot_cfg = YellowbrickPlotConfig(
            experiment=self._make_regression_experiment(),
            plot_type="learning_curve",
            plot_params={},
        )

        cv = plot_cfg.parse_cv()

        assert isinstance(cv, KFold)
        assert cv.n_splits == 5

    def test_parse_cv_uses_explicit_integer(self):
        plot_cfg = YellowbrickPlotConfig(
            experiment=self._make_classification_experiment(),
            plot_type="learning_curve",
            plot_params={"cv": 3},
        )

        cv = plot_cfg.parse_cv()

        assert isinstance(cv, StratifiedKFold)
        assert cv.n_splits == 3

    def test_one_classification_plot(self):
        experiment = self._make_classification_experiment()
        experiment()
        plot_type = "roc_auc"  # Example classification plot
        plot_cfg = YellowbrickPlotConfig(
            experiment=experiment,
            plot_type=plot_type,
            features="all",
            classes="all",
            title=plot_type.replace("_", " ").title(),
            save_path=f"{self.temp_dir}/{plot_type}_dataconfig.png",
        )
        plot_cfg()
        assert Path(f"{self.temp_dir}/{plot_type}_dataconfig.png").exists()

    @classmethod
    def setup_class(cls):
        config_dir = (
            Path(__file__).resolve().parents[3] / "examples" / "sklearn" / "config"
        )
        config_dir = Path(config_dir).resolve().as_posix() + "/"
        cls.classification_data_config = config_dir + "data/classification.yaml"
        cls.classification_model_config = config_dir + "model/logistic.yaml"
        cls.regression_data_config = config_dir + "data/regression.yaml"
        cls.regression_model_config = config_dir + "model/ridge.yaml"
        cls.cluster_data_config = config_dir + "data/cluster.yaml"
        cls.cluster_model_config = config_dir + "model/kmeans.yaml"
        cls.temp_dir = mkdtemp()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.temp_dir)

    def test_one_regression_plot(self):
        regression_files = FileConfig(data_file="", model_file="")
        regression_data = OmegaConf.load(self.regression_data_config)
        regression_model = OmegaConf.load(self.regression_model_config)
        experiment = ExperimentConfig(
            data=regression_data,
            model=regression_model,
            files=regression_files,
            classifier=False,
        )
        plot_type = "prediction_error"  # Example regression plot
        plot_cfg = YellowbrickPlotConfig(
            experiment=experiment,
            plot_type=plot_type,
            features="all",
            classes="all",
            title=plot_type.replace("_", " ").title() + " (Regression)",
            save_path=f"{self.temp_dir}/{plot_type}_regression.png",
        )
        plot_cfg()
        assert Path(f"{self.temp_dir}/{plot_type}_regression.png").exists()

    def test_clustering_plots(self):
        experiment = self._make_clustering_experiment(persisted_files=True)
        for plot_type in cluster_viz_types:
            if plot_type in ["intercluster_distance"]:
                continue
            else:
                filepath = f"{self.temp_dir}/{plot_type}_clustering.png"
                plot_cfg = YellowbrickPlotConfig(
                    experiment=experiment,
                    plot_type=plot_type,
                    features="all",
                    classes="all",
                    title=plot_type.replace("_", " ").title() + " (Clustering)",
                    save_path=filepath,
                )
                plot_cfg()
                assert Path(filepath).exists()

            assert Path(f"{self.temp_dir}/k_elbow_clustering.png").exists()

    def test_single_plot_prepares_experiment_only_once(self):
        plot_cfg = YellowbrickPlotConfig(
            experiment=self._make_classification_experiment_with_persisted_files(
                "single_prepare",
            ),
            plot_type="roc_auc",
            save_path=f"{self.temp_dir}/single_prepare.png",
        )

        with (
            patch.object(
                ExperimentConfig,
                "__call__",
                autospec=True,
                return_value={"accuracy": 0.9},
            ) as mock_experiment_call,
            patch.object(
                YellowbrickPlotConfig,
                "visualize",
                autospec=True,
                return_value=None,
            ),
            patch.object(
                plot_cfg,
                "_experiment_outputs_ready",
                side_effect=[False, False, True, True, True],
            ),
        ):
            first_scores = plot_cfg()
            second_scores = plot_cfg()

        assert mock_experiment_call.call_count == 1
        assert first_scores == {"accuracy": 0.9}
        assert second_scores == {"accuracy": 0.9}

    def test_single_plot_applies_rc_config(self):
        plot_cfg = YellowbrickPlotConfig(
            experiment=self._make_classification_experiment_with_persisted_files(
                "rc_single",
            ),
            plot_type="roc_auc",
            rc_config={"figure.figsize": (7, 5)},
            save_path=f"{self.temp_dir}/rc_single.png",
        )

        with (
            patch(
                "deckard.plugins.yellowbrick.plot.plt.rcParams.update",
            ) as mock_rc_update,
            patch.object(
                YellowbrickPlotConfig,
                "_ensure_experiment_prepared",
                return_value={},
            ),
            patch.object(YellowbrickPlotConfig, "visualize", return_value=None),
        ):
            plot_cfg()

        mock_rc_update.assert_called_once_with({"figure.figsize": (7, 5)})

    def test_plot_list_applies_rc_config(self):
        plot_cfg = YellowbrickConfigList(
            experiment=self._make_classification_experiment_with_persisted_files(
                "rc_list",
            ),
            plots=["roc_auc"],
            rc_config={"figure.figsize": (8, 6)},
            plot_folder=self.temp_dir,
        )

        with (
            patch(
                "deckard.plugins.yellowbrick.plot.plt.rcParams.update",
            ) as mock_rc_update,
            patch.object(
                YellowbrickConfigList,
                "_ensure_experiment_prepared",
                return_value={},
            ),
            patch.object(
                YellowbrickConfigList,
                "_set_plot_dict",
                return_value=None,
            ),
        ):
            plot_cfg()

        mock_rc_update.assert_called_once_with({"figure.figsize": (8, 6)})
