import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from dvc.api import params_show
# from dvc.api import open
from sklearn.datasets import make_classification, make_regression
from yellowbrick.features import RadViz, Rank1D, Rank2D
from yellowbrick.target import ClassBalance, FeatureCorrelation
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    params = params_show()['data']
    if "sample_size" in params:
        sample_size = params.pop("sample_size")
    else:
        sample_size = .8
    if "classification" in list(params):
        class_params = params.pop("classification")
        X_train, y_train = make_classification(**class_params)
    elif "regression" in list(params):
        regression_params = params.pop("regression")
        X_train, y_train = make_regression(**regression_params)
    else:
        raise NotImplementedError("Only classification is implemented")
    length = int(X_train.shape[0]/2)
    X_test, y_test = X_train[:int(length * sample_size)], y_train[:int(length * sample_size)]
    classes = set(y_train)
    features = X_train.shape[1]
    if "add_noise" in list(params):
        noise_function = np.random.normal
        if "X_train" in params["add_noise"]:
            if not isinstance(params["add_noise"]["X_train"], list):
                params['add_noise']['X_train'] = [params['add_noise']['X_train']]
            X_train += noise_function(*params["add_noise"]["X_train"])
        if "X_test" in params["add_noise"]:
            if not isinstance(params["add_noise"]["X_test"], list):
                params['add_noise']['X_test'] = [params['add_noise']['X_test']]
            X_test += noise_function(*params["add_noise"]["X_test"])
        params.pop("add_noise")
    file_params = params.pop("files")
    plots = params.pop("plots")
    assert params == {}, f"Unrecognized parameters {params}"
    params = file_params 
    assert "path" in params, "Path to save data is not specified"
    assert "file" in params, "File name is not specified"
    Path(params["path"]).mkdir(parents=True, exist_ok=True)
    np.savez(str(Path(params['path'], params["file"])), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    params.pop("file")
    X_train = X_train[:2000]
    y_train = y_train[:2000]
    if "ground_truth" in list(params):
        with open(Path(params['path'], params["ground_truth"]), "w") as f:
            json.dump(y_train.tolist(), f)
        params.pop("ground_truth")
    if len(plots.keys()) > 0:
        assert "path" in plots, "Path to save plots is not specified"
        Path(plots["path"]).mkdir(parents=True, exist_ok=True)
        if "radviz" in plots:
            visualizer = RadViz(classes = classes)
            visualizer.fit(X_train, y_train)
            visualizer.show(Path(plots['path'],plots["radviz"]))
            plots.pop("radviz")
            plt.gcf().clear()
        if "rank1d" in plots:
            visualizer = Rank1D(algorithm = "shapiro")
            visualizer.fit(X_train, y_train)
            visualizer.show(Path(plots['path'],plots["rank1d"]))
            plots.pop("rank1d")
            plt.gcf().clear()
        if "rank2d" in plots:
            visualizer = Rank2D(algorithm = "pearson")
            visualizer.fit(X_train, y_train)
            visualizer.show(Path(plots['path'],plots["rank2d"]))
            plots.pop("rank2d")
            plt.gcf().clear()
        if "balance" in plots:
            visualizer = ClassBalance(labels = classes)
            visualizer.fit(y_train)
            visualizer.show(Path(plots['path'],plots["balance"]))
            plots.pop("balance")
            plt.gcf().clear()
        if "correlation" in plots:
            visualizer = FeatureCorrelation(labels = list(range(features)))
            visualizer.fit(X_train, y_train)
            visualizer.show(Path(plots['path'],plots["correlation"]))
            plots.pop("correlation")
            plt.gcf().clear()
    params.pop("path")
    assert params == {}, f"Unrecognized parameters {params}"
        
    
    
