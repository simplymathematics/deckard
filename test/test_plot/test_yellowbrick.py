import unittest
from pathlib import Path
from tempfile import mkdtemp
import shutil
from unittest.mock import patch
from omegaconf import OmegaConf
import pytest
from deckard.experiment import ExperimentConfig
from deckard.file import FileConfig
from sklearn.datasets import make_classification

pytest.importorskip("yellowbrick")
from deckard.plot.yellowbrick_plots import (  # NOQA E402
    YellowbrickConfigList,
    YellowbrickPlotConfig,
    model_selection_viz_types,
    cluster_viz_types,
)
from deckard.data import DataConfig  # noqa 402
from deckard.model import ModelConfig  # noqa 402

expensive_viz_types = [
    "manifold",
    "rfecv",
    "validation_curve",
    "learning_curve",
    "dropping_curve",
    "intercluster_distance",
    "feature_importances",
    "cv_scores",
    "silhouette",
] + model_selection_viz_types


class TestYellowbrickPlots(unittest.TestCase):
    def test_one_classification_plot(self):
        files = FileConfig(data_file="", model_file="")
        data_conf = OmegaConf.load(self.classification_data_config)
        model_conf = OmegaConf.load(self.classification_model_config)
        data = DataConfig(**OmegaConf.to_container(data_conf, resolve=True))
        model = ModelConfig(**OmegaConf.to_container(model_conf, resolve=True))
        experiment = ExperimentConfig(data=data, model=model, files=files)
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
        self.assertTrue(
            Path(f"{self.temp_dir}/{plot_type}_dataconfig.png").exists(),
        )

    @classmethod
    def setUpClass(cls):
        this_file = Path(__file__)
        config_dir = (
            this_file.parent.parent.parent / "examples" / "sklearn" / "config"
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
    def tearDownClass(cls):
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
        self.assertTrue(
            Path(f"{self.temp_dir}/{plot_type}_regression.png").exists(),
        )

    def test_one_clustering_plot(self):
        cluster_files = FileConfig(data_file="", model_file="")
        cluster_data = OmegaConf.load(self.cluster_data_config)
        cluster_model = OmegaConf.load(self.cluster_model_config)
        cluster_model["scorer"] = None
        experiment = ExperimentConfig(
            data=cluster_data,
            model=cluster_model,
            files=cluster_files,
        )
        plot_type = "k_elbow"  # Example clustering plot
        plot_cfg = YellowbrickPlotConfig(
            experiment=experiment,
            plot_type=plot_type,
            features="all",
            classes="all",
            title=plot_type.replace("_", " ").title() + " (Clustering)",
            save_path=f"{self.temp_dir}/{plot_type}_clustering.png",
        )
        plot_cfg()
        self.assertTrue(
            Path(f"{self.temp_dir}/{plot_type}_clustering.png").exists(),
        )

    def test_clustering_plots(self):
        cluster_data_file = f"{self.temp_dir}/data/cluster_data.pkl"
        cluster_model_file = f"{self.temp_dir}/models/kmeans_model.pkl"
        cluster_files = FileConfig(
            data_file=cluster_data_file,
            model_file=cluster_model_file,
        )
        cluster_data = OmegaConf.load(self.cluster_data_config)
        cluster_model = OmegaConf.load(self.cluster_model_config)
        cluster_model["scorer"] = None
        experiment = ExperimentConfig(
            data=cluster_data,
            model=cluster_model,
            files=cluster_files,
        )
        for plot_type in cluster_viz_types:
            if plot_type in ["intercluster_distance"]:
                continue
            else:
                filepath = f"{self.temp_dir}/{plot_type}_clustering.png"
                with self.subTest(plot_type=plot_type):
                    plot_cfg = YellowbrickPlotConfig(
                        experiment=experiment,
                        plot_type=plot_type,
                        features="all",
                        classes="all",
                        title=plot_type.replace("_", " ").title()
                        + " (Clustering)",
                        save_path=filepath,
                    )
                    plot_cfg()
                    self.assertTrue(Path(filepath).exists())

    def test_single_plot_prepares_experiment_only_once(self):
        classification_data = OmegaConf.load(self.classification_data_config)
        classification_model = OmegaConf.load(self.classification_model_config)
        files = FileConfig(
            data_file=f"{self.temp_dir}/data/single_prepare.pkl",
            model_file=f"{self.temp_dir}/models/single_prepare.pkl",
        )
        plot_cfg = YellowbrickPlotConfig(
            experiment=ExperimentConfig(
                data=classification_data,
                model=classification_model,
                files=files,
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

        self.assertEqual(mock_experiment_call.call_count, 1)
        self.assertEqual(first_scores, {"accuracy": 0.9})
        self.assertEqual(second_scores, {"accuracy": 0.9})

    def test_plot_list_reuses_prepared_experiment_for_children(self):
        classification_data = OmegaConf.load(self.classification_data_config)
        classification_model = OmegaConf.load(self.classification_model_config)
        files = FileConfig(
            data_file=f"{self.temp_dir}/data/list_prepare.pkl",
            model_file=f"{self.temp_dir}/models/list_prepare.pkl",
        )
        _ = YellowbrickConfigList(
            experiment=ExperimentConfig(
                data=classification_data,
                model=classification_model,
                files=files,
            ),
            plots=["roc_auc", "precision_recall_curve"],
            plot_folder=self.temp_dir,
        )

    def test_single_plot_applies_rc_config(self):
        # Setup minimal classification data and model
        Xc, yc = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42,
        )
        data = OmegaConf.create(
            {
                "X_train": Xc.tolist(),
                "y_train": yc.tolist(),
                "X_test": Xc.tolist(),
                "y_test": yc.tolist(),
            },
        )
        from deckard.model import ModelConfig

        model = ModelConfig(
            model_type="sklearn.linear_model.LogisticRegression",
        )
        files = FileConfig(
            data_file=f"{self.temp_dir}/data/rc_single.pkl",
            model_file=f"{self.temp_dir}/models/rc_single.pkl",
        )
        plot_cfg = YellowbrickPlotConfig(
            experiment=ExperimentConfig(
                data=data,
                model=model,
                files=files,
            ),
            plot_type="roc_auc",
            rc_config={"figure.figsize": (7, 5)},
            save_path=f"{self.temp_dir}/rc_single.png",
        )

        with (
            patch(
                "deckard.plot.yellowbrick_plots.plt.rcParams.update",
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
        classification_data = OmegaConf.load(self.classification_data_config)
        classification_model = OmegaConf.load(self.classification_model_config)
        files = FileConfig(
            data_file=f"{self.temp_dir}/data/rc_list.pkl",
            model_file=f"{self.temp_dir}/models/rc_list.pkl",
        )
        plot_cfg = YellowbrickConfigList(
            experiment=ExperimentConfig(
                data=classification_data,
                model=classification_model,
                files=files,
            ),
            plots=["roc_auc"],
            rc_config={"figure.figsize": (8, 6)},
            plot_folder=self.temp_dir,
        )

        with (
            patch(
                "deckard.plot.yellowbrick_plots.plt.rcParams.update",
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


if __name__ == "__main__":
    unittest.main()
