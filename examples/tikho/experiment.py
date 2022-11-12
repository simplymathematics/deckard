import collections
from pathlib import Path
from time import process_time

import matplotlib.pyplot as plt
import numpy as np
import yaml
from dvc.api import params_show
from dvclive import Live
from tqdm import tqdm
from yellowbrick.classifier import classification_report, confusion_matrix
from yellowbrick.contrib.wrapper import classifier
from yellowbrick.features import (
    PCA,
    Manifold,
    ParallelCoordinates,
    RadViz,
    Rank1D,
    Rank2D,
)
from yellowbrick.target import (
    BalancedBinningReference,
    ClassBalance,
    # FeatureCorrelation,
)

from data import Data
from json_mixin import JSONMixin
from model import Model

target_visualizers = {
    "bins": BalancedBinningReference,
    "class_balance": ClassBalance,
    # "correlation" : FeatureCorrelation,
}

feature_visualizers = {
    "radviz": RadViz,
    "pca": PCA,
    "manifold": Manifold,
    "rank1d": Rank1D,
    "rank2d": Rank2D,
    "parallel": ParallelCoordinates,
}

classification_visualizers = {
    "confusion": confusion_matrix,
    "classification": classification_report,
}


class Experiment(
    collections.namedtuple(
        typename="Experiment",
        field_names="data, model, is_fitted, scorers, plots, files, fit",
        defaults=({}, {}, {}, {}),
    ),
    JSONMixin,
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    def fit(self, data: dict, model: object) -> tuple:
        """
        Fits model to data.
        :param data: dict, data to fit model to.
        :param model: object, model to fit.
        :returns: tuple, (model, fit_time).
        """
        if not hasattr(data, "X_train"):
            data = data.load()
            assert hasattr(data, "X_train"), "Data must have X_train"
        if not hasattr(model, "fit"):
            model = model.load()
        start = process_time()
        result = model.fit(data.X_train, data.y_train)
        result = process_time() - start
        return model, result / len(data.X_train)

    def predict(self, data: dict, model: object) -> tuple:
        """
        Predicts data with model.
        :param data: dict, data to predict.
        :param model: object, model to predict with.
        :returns: tuple, (predictions, predict_time).
        """
        start = process_time()
        if not hasattr(model, "predict"):
            model = model.load()
        predictions = model.predict(data.X_test)
        result = process_time() - start
        return predictions, result / len(data.X_test)

    def load(config: str) -> tuple:
        """
        Loads data, model from the config file.
        :param config: str, path to config file.
        :returns: tuple(dict, object), (data, model).
        """
        yaml.add_constructor("!Data:", Data)
        yaml.add_constructor("!Model:", Model)
        data_document = """!Data:\n""" + str(dict(experiment.data))
        model_document = """!Model:\n""" + str(dict(experiment.model))
        data = yaml.load(data_document, Loader=yaml.Loader)
        data = data.load()
        model = yaml.load(model_document, Loader=yaml.Loader)
        model = model.load()
        return data, model


if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--stage", default="classify")
    params = params_show("params.yaml")
    files = params["files"]
    epochs = params["fit"]["epochs"]
    log_interval = params["fit"]["log_interval"]
    learning_rate = params["fit"]["learning_rate"]
    yaml.add_constructor("!Experiment:", Experiment)
    experiment = yaml.load("!Experiment:\n" + str(params), Loader=yaml.Loader)
    data, model = Experiment.load(experiment)
    logger = Live(path=Path(files["path"]), report="html")
    logger.log_params(params)
    for i in tqdm(range(epochs)):
        start = process_time()
        model = model.fit(
            data.X_train,
            data.y_train,
            learning_rate=learning_rate,
            epochs=1,
        )
        if i % log_interval == 0:
            fit_time = process_time() - start
            logger.log("time", fit_time)
            train_score = model.score(data.X_train, data.y_train)
            test_score = model.score(data.X_test, data.y_test)
            logger.log("train_score", train_score)
            logger.log("loss", model.loss(data.X_train, data.y_train))
            logger.log("weights", np.mean(model.weights))
            logger.log("bias", np.mean(model.bias))
            logger.log("epoch", i * log_interval)
            logger.next_step()

    predictions = model.predict(data.X_test)
    proba = model.predict_proba(data.X_test)
    predictions, predict_time = experiment.predict(data, model)
    ground_truth = data.y_test
    time_dict = {"fit_time": fit_time, "predict_time": predict_time}
    score_dict = experiment.score(data, predictions)
    output_files = experiment.save(
        **experiment.files,
        data=data,
        model=model,
        ground_truth=ground_truth,
        predictions=predictions,
        time_dict=time_dict,
        score_dict=score_dict,
        params=params,
    )
    for file in output_files:
        assert file.exists(), f"File {file} does not exist."
    plots = params.pop("plots")
    yb_model = classifier(model)
    path = files.pop("path")
    for name in classification_visualizers.keys():
        visualizer = classification_visualizers[name]
        viz = visualizer(
            yb_model,
            X_train=data.X_train,
            y_train=data.y_train,
            classes=[int(y) for y in np.unique(data.y_train)],
        )
        viz.show(outpath=Path(path, name))
        # logger.log_image(name, Path(path, name))
        plt.gcf().clear()
        i += 1
