import os
from pathlib import Path
from tempfile import mkdtemp

import matplotlib
import numpy as np
import pytest
from omegaconf import OmegaConf
from sklearn.model_selection import KFold, ShuffleSplit, TimeSeriesSplit

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)

pytest.importorskip("yellowbrick")

from deckard.data import DataConfig
from deckard.experiment import ExperimentConfig
from deckard.file import FileConfig
from deckard.model import ModelConfig
from deckard.plugins.yellowbrick import plot as yb_plot


class _ModelWithName:
    pass


class _PredictConfig:
    def __init__(self, output, *, library="sklearn", classifier=True):
        self._output = output
        self.library = library
        self.classifier = classifier
        self.device = "cpu"

    def predict(self, X):
        _ = X
        return self._output

    def get_model(self):
        return _ModelWithName()


class _DummyVisualizer:
    def __init__(self, *args, ax=None, axes=None, **kwargs):
        _ = (args, kwargs)
        self.ax = ax
        self.axes = axes
        self.classes_ = None

    def fit(self, *args, **kwargs):
        _ = (args, kwargs)
        return self

    def transform(self, *args, **kwargs):
        _ = (args, kwargs)
        return self

    def fit_transform(self, *args, **kwargs):
        _ = (args, kwargs)
        return self

    def score(self, *args, **kwargs):
        _ = (args, kwargs)
        return 1.0

    def finalize(self):
        return None

    def show(self, *args, **kwargs):
        _ = (args, kwargs)
        return None


@pytest.fixture(scope="module")
def temp_dir():
    path = mkdtemp()
    yield path
    Path(path).mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def experiments(temp_dir):
    config_dir = (
        Path(__file__).resolve().parents[3] / "examples" / "sklearn" / "config"
    )
    data_conf = OmegaConf.load(
        (config_dir / "data" / "classification.yaml").as_posix()
    )
    model_conf = OmegaConf.load((config_dir / "model" / "logistic.yaml").as_posix())
    reg_data_conf = OmegaConf.load(
        (config_dir / "data" / "regression.yaml").as_posix()
    )
    reg_model_conf = OmegaConf.load((config_dir / "model" / "ridge.yaml").as_posix())

    cls_exp = ExperimentConfig(
        data=DataConfig(**OmegaConf.to_container(data_conf, resolve=True)),
        model=ModelConfig(**OmegaConf.to_container(model_conf, resolve=True)),
        files=FileConfig(data_file="", model_file=""),
    )
    reg_exp = ExperimentConfig(
        data=DataConfig(**OmegaConf.to_container(reg_data_conf, resolve=True)),
        model=ModelConfig(**OmegaConf.to_container(reg_model_conf, resolve=True)),
        files=FileConfig(data_file="", model_file=""),
        classifier=False,
    )
    return {"classifier": cls_exp, "regressor": reg_exp}


def _make_plot_cfg(experiment, plot_type="roc_auc", plot_params=None):
    return yb_plot.YellowbrickPlotConfig(
        experiment=experiment,
        plot_type=plot_type,
        plot_params=plot_params or {},
        save_path="/tmp/yellowbrick_unit_plot.png",
    )


def test_get_shape_basic_and_nested_dataset_forms():
    arr = np.zeros((3, 2))
    assert yb_plot._get_shape(arr) == (3, 2)

    class _WithDataset:
        def __init__(self):
            self.dataset = np.zeros((10, 4))
            self.indices = [0, 1, 2]

    assert yb_plot._get_shape(_WithDataset()) == (3, 4)


def test_get_shape_sequence_and_missing_shape_error():
    seq = [(np.array([1.0, 2.0]), 0), (np.array([3.0, 4.0]), 1)]
    assert yb_plot._get_shape(seq) == (2, 2)

    with pytest.raises(AttributeError):
        yb_plot._get_shape(object())


def test_named_classifier_adapter_preserves_model_name():
    adapter = yb_plot._named_classifier_adapter(_PredictConfig(np.array([0.0, 1.0])))
    assert adapter.__class__.__name__ == "_ModelWithName"


def test_adapter_predict_proba_predict_and_score_variants():
    logits_adapter = yb_plot._YellowbrickModelAdapter(
        _PredictConfig(np.array([-1.0, 1.0]))
    )
    probs = logits_adapter.predict_proba(np.array([[0.0], [1.0]]))
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    multiclass_adapter = yb_plot._YellowbrickModelAdapter(
        _PredictConfig(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])),
    )
    labels = multiclass_adapter.predict(np.array([[0.0], [1.0]]))
    assert labels.tolist() == [2, 0]

    empty_score_adapter = yb_plot._YellowbrickModelAdapter(
        _PredictConfig(np.array([]))
    )
    assert empty_score_adapter.score(np.array([]), np.array([])) == 0.0


def test_adapter_raises_for_unsupported_prediction_shape():
    adapter = yb_plot._YellowbrickModelAdapter(_PredictConfig(np.zeros((2, 2, 2))))
    with pytest.raises(ValueError):
        adapter.predict_proba(np.array([[0.0], [1.0]]))
    with pytest.raises(ValueError):
        adapter.predict(np.array([[0.0], [1.0]]))


def test_parse_cv_dict_variants_and_invalid_name(experiments):
    cfg = _make_plot_cfg(
        experiments["classifier"],
        plot_type="validation_curve",
        plot_params={"cv": {"name": "timeseries", "n_splits": 3}},
    )
    assert isinstance(cfg.parse_cv(), TimeSeriesSplit)

    cfg = _make_plot_cfg(
        experiments["classifier"],
        plot_type="validation_curve",
        plot_params={"cv": {"name": "shufflesplit", "n_splits": 3, "test_size": 0.2}},
    )
    assert isinstance(cfg.parse_cv(), ShuffleSplit)

    cfg = _make_plot_cfg(
        experiments["regressor"],
        plot_type="validation_curve",
        plot_params={"cv": {"name": "kfold", "n_splits": 2}},
    )
    assert isinstance(cfg.parse_cv(), KFold)

    cfg = _make_plot_cfg(
        experiments["classifier"],
        plot_type="validation_curve",
        plot_params={"cv": {"name": "unknown", "n_splits": 3}},
    )
    with pytest.raises(ValueError, match="Unsupported CV type"):
        cfg.parse_cv()


def test_parse_range_variants_and_invalid_distribution(experiments):
    cfg = _make_plot_cfg(
        experiments["classifier"],
        plot_type="validation_curve",
        plot_params={"param_range": (0.1, 1.0), "num": 3},
    )
    assert cfg.parse_range() == [0.1, 0.55, 1.0]

    cfg = _make_plot_cfg(
        experiments["classifier"],
        plot_type="validation_curve",
        plot_params={"param_range": [1, 100, "log"], "num": 3},
    )
    assert cfg.parse_range() == [1.0, 10.0, 100.0]

    cfg = _make_plot_cfg(
        experiments["classifier"],
        plot_type="validation_curve",
        plot_params={"param_range": [0, 6, 2]},
    )
    assert cfg.parse_range() == [0, 3, 6]

    cfg = _make_plot_cfg(
        experiments["classifier"],
        plot_type="validation_curve",
        plot_params={"param_range": [1, 2, "bad"]},
    )
    with pytest.raises(ValueError, match="Distribution"):
        cfg.parse_range()


def test_visualize_feature_target_regressor_classifier_cluster_model_selection_branches(
    monkeypatch,
    experiments,
):
    for name in (
        "Rank1D",
        "Rank2D",
        "RadViz",
        "ParallelCoordinates",
        "JointPlotVisualizer",
        "PCADecomposition",
        "Manifold",
        "ClassBalance",
        "BalancedBinningReference",
        "FeatureCorrelation",
        "PredictionError",
        "ResidualsPlot",
        "ManualAlphaSelection",
        "ROCAUC",
        "PrecisionRecallCurve",
        "ClassificationReport",
        "ClassPredictionError",
        "DiscriminationThreshold",
        "KElbowVisualizer",
        "SilhouetteVisualizer",
        "InterclusterDistance",
        "ValidationCurve",
        "LearningCurve",
        "CVScores",
        "FeatureImportances",
        "RFECV",
        "DroppingCurve",
    ):
        monkeypatch.setattr(yb_plot, name, _DummyVisualizer)

    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 1] * 5)

    def _stub_plot_data(test=False, attack=False):
        _ = attack
        if test:
            return X, y, [0, 1], [0, 1]
        return X, y, [0, 1], [0, 1]

    feature_types = [
        "rank1d",
        "rank2d",
        "radviz",
        "pcoords",
        "jointplot",
        "pca",
        "manifold",
    ]
    target_types = [
        "class_balance",
        "balanced_binning_reference",
        "feature_correlation",
    ]
    reg_types = ["prediction_error", "residuals_plot", "alpha_selection"]
    cls_types = [
        "roc_auc",
        "precision_recall_curve",
        "classification_report",
        "class_prediction_error",
        "discrimination_threshold",
    ]
    cluster_types = ["k_elbow", "silhouette", "intercluster_distance"]
    model_sel_types = [
        "validation_curve",
        "learning_curve",
        "cv_scores",
        "feature_importances",
        "rfecv",
        "dropping_curve",
    ]

    for plot_type in (
        feature_types
        + target_types
        + reg_types
        + cls_types
        + cluster_types
        + model_sel_types
    ):
        cfg = _make_plot_cfg(experiments["classifier"], plot_type=plot_type)
        monkeypatch.setattr(cfg, "_get_plot_data", _stub_plot_data)
        monkeypatch.setattr(cfg, "_get_plot_model", lambda: object())
        monkeypatch.setattr(cfg, "show", lambda _visualizer: None)
        if plot_type == "jointplot":
            cfg.plot_params = {"columns": [0, 1]}
        elif plot_type in {"validation_curve", "learning_curve"}:
            cfg.plot_params = {
                "cv": {"name": "kfold", "n_splits": 2},
                "param_range": [1, 3, "linear"],
                "num": 3,
            }
        elif plot_type in {
            "cv_scores",
            "feature_importances",
            "rfecv",
            "dropping_curve",
        }:
            cfg.plot_params = {"cv": {"name": "kfold", "n_splits": 2}}
        else:
            cfg.plot_params = {}

        cfg.visualize(ax=object())


def test_show_sets_title_and_saves_file(monkeypatch, experiments, tmp_path):
    cfg = _make_plot_cfg(experiments["classifier"], plot_type="roc_auc")
    save_path = (tmp_path / "yb_show.png").as_posix()
    cfg.save_path = save_path
    cfg.title = "My Plot"

    class _Axes:
        def __init__(self):
            self._title = ""

        def set_title(self, title):
            self._title = title

    class _Visual(_DummyVisualizer):
        def __init__(self):
            super().__init__(ax=_Axes())

    monkeypatch.setattr(yb_plot, "all_viz_objects", [_Visual])
    saved = []
    monkeypatch.setattr(yb_plot.plt, "savefig", lambda path: saved.append(path))

    visualizer = _Visual()
    cfg.show(visualizer)

    assert visualizer.ax._title == "My Plot"
    assert saved == [save_path]


def test_config_list_set_plot_dict_and_len(monkeypatch, experiments, tmp_path):
    exp = experiments["classifier"]
    exp.data.X_train = np.arange(20).reshape(10, 2)
    exp.data._X = exp.data.X_train
    exp.data._y = np.array([0, 1] * 5)
    cfg = yb_plot.YellowbrickConfigList(
        experiment=exp,
        plots=["roc_auc", "precision_recall_curve"],
        plot_folder=tmp_path.as_posix(),
    )

    cfg._set_plot_dict()

    assert len(cfg) == 2
    assert set(cfg.plots.keys()) == {"roc_auc", "precision_recall_curve"}

    monkeypatch.setattr(cfg, "_ensure_experiment_prepared", lambda: {"ok": 1})
    monkeypatch.setattr(
        yb_plot.YellowbrickConfigList, "_set_plot_dict", lambda self: None
    )
    monkeypatch.setattr(
        yb_plot.YellowbrickPlotConfig, "__call__", lambda self: {"ok": 1}
    )
    assert cfg() == {"ok": 1}
