from yellowbrick.classifier import (classification_report, confusion_matrix, roc_auc, )
from yellowbrick.contrib.wrapper import classifier
# from yellowbrick.exceptions import ModelError
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from yellowbrick.features import RadViz, Rank1D, Rank2D, PCA, Manifold, ParallelCoordinates
from yellowbrick.target import ClassBalance, FeatureCorrelation
from argparse import Namespace


classification_visualizers = {
    "confusion": confusion_matrix,
    "classification": classification_report,
    "roc_auc": roc_auc
}



def visualise_classification(plots: dict, data:Namespace, model:object, path:str) -> list:
    """
    Visualise classification results according to the plots dictionary.
    :param plots: dict of plots to be generated
    :param data: dict of data to be used for visualisation
    :param model: model object
    :param path: path to save the plots
    :return: list of paths to the generated plots
    """
    paths = []
    yb_model = classifier(model)        
    for name in classification_visualizers.keys():
        if name in plots.keys():
            visualizer = classification_visualizers[name]
            if len(set(data.y_train)) > 2:
                viz = visualizer(
                    yb_model,
                    X_train=data.X_train,
                    y_train=data.y_train,
                    classes=[int(y) for y in np.unique(data.y_train)],
                )
            elif len(set(data.y_train)) == 2:
                viz = visualizer(
                    yb_model,
                    X_train=data.X_train,
                    y_train=data.y_train,
                    binary = True
                )
            else:
                viz = visualizer(
                    yb_model,
                    X_train=data.X_train,
                    y_train=data.y_train,
                    classes=[0],
                )
            viz.show(outpath=Path(path, name))
            assert Path(path, str(name)+".png").is_file(), f"File {name} does not exist."
            paths.append(str(Path(path, str(name)+".png")))
            plt.gcf().clear()
    return paths



def visualise_data( data: Namespace, files: dict, plots: dict, classes:list = None, features:list = None) -> list:
        y_train = data.y_train
        X_train = data.X_train
        classes = set(y_train) if classes is None else classes
        features = X_train.shape[1] if features is None else features
        paths = []
        if len(plots.keys()) > 0:
            assert (
                "path" in files
            ), "Path to save plots is not specified in the files entry in params.yaml"
            Path(files["path"]).mkdir(parents=True, exist_ok=True)
            if "radviz" in plots:
                visualizer = RadViz(classes=classes)
                visualizer.fit(X_train, y_train)
                visualizer.show(Path(files["path"], plots["radviz"]))
                paths.append(Path(files["path"], plots["radviz"]))
                plots.pop("radviz")
                plt.gcf().clear()
            if "rank1d" in plots:
                visualizer = Rank1D(algorithm="shapiro")
                visualizer.fit(X_train, y_train)
                visualizer.show(Path(files["path"], plots["rank1d"]))
                paths.append(Path(files["path"], plots["rank1d"]))
                plots.pop("rank1d")
                plt.gcf().clear()
            if "rank2d" in plots:
                visualizer = Rank2D(algorithm="pearson")
                visualizer.fit(X_train, y_train)
                visualizer.show(Path(files["path"], plots["rank2d"]))
                paths.append(Path(files["path"], plots["rank2d"]))
                plots.pop("rank2d")
                plt.gcf().clear()
            if "balance" in plots:
                visualizer = ClassBalance(labels=classes)
                visualizer.fit(y_train)
                visualizer.show(Path(files["path"], plots["balance"]))
                paths.append(Path(files["path"], plots["balance"]))
                plots.pop("balance")
                plt.gcf().clear()
            if "correlation" in plots:
                visualizer = FeatureCorrelation(labels=list(range(features)))
                visualizer.fit(X_train, y_train)
                visualizer.show(Path(files["path"], plots["correlation"]))
                paths.append(Path(files["path"], plots["correlation"]))
                plots.pop("correlation")
                plt.gcf().clear()
            if "pca" in plots:
                visualizer = PCA()
                visualizer.fit_transform(X_train, y_train)
                visualizer.show(Path(files["path"], plots["pca"]))
                paths.append(Path(files["path"], plots["pca"]))
                plots.pop("pca")
                plt.gcf().clear()
            if "manifold" in plots:
                visualizer = Manifold(manifold="tsne")
                visualizer.fit_transform(X_train, y_train)
                visualizer.show(Path(files["path"], plots["manifold"]))
                paths.append(Path(files["path"], plots["manifold"]))
            if "parallel" in plots:
                visualizer = ParallelCoordinates(classes = classes, features = list(range(features)))
                visualizer.fit(X_train, y_train)
                visualizer.show(Path(files["path"], plots["parallel"]))
                paths.append(Path(files["path"], plots["parallel"]))
                plots.pop("parallel")
                plt.gcf().clear()
        return paths
