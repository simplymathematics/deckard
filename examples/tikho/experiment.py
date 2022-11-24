import collections
from time import process_time

import yaml
from dvc.api import params_show
from yellowbrick.classifier import (
    classification_report,
    confusion_matrix,
    roc_auc,
)
from data import Data
from model import Model

from .hashable import BaseHashable

classification_visualizers = {
    "confusion": confusion_matrix,
    "classification": classification_report,
    "roc_auc": roc_auc,
}


class Experiment(
    collections.namedtuple(
        typename="Experiment",
        field_names="data, model, scorers, plots, files, fit, predict",
        defaults=({}, {}, {}, {}, {}, {}),
    ),
    BaseHashable,
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

    def load(self) -> tuple:
        """
        Loads data, model from the config file.
        :param config: str, path to config file.
        :returns: tuple(dict, object), (data, model).
        """

        yaml.add_constructor("!Data:", Data)
        data_document = """!Data:\n""" + str(dict(self.data))
        data = yaml.load(data_document, Loader=yaml.Loader)
        data = data.load()
        if self.model is not None:
            yaml.add_constructor("!Model:", Model)
            model_document = """!Model:\n""" + str(dict(self.model))
            model = yaml.load(model_document, Loader=yaml.Loader)
            model = model.load()
        return data, model


yaml.add_constructor("!Experiment:", Experiment)

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

    experiment = yaml.load("!Experiment:\n" + str(params), Loader=yaml.Loader)
    data, model = Experiment.load(experiment)
    # logger = Live(path=Path(files["path"]), report="html")
    # logger.log_params(params)
    # epochs = int(round(epochs//log_interval))
    # for i in tqdm(range(epochs)):
    #     start = process_time()
    #     model = model.fit(
    #         data.X_train,
    #         data.y_train,
    #         learning_rate=learning_rate,
    #         epochs=1,
    #     )
    #     if i % log_interval == 0:
    #         fit_time = process_time() - start
    #         logger.log("time", fit_time)
    #         train_score = model.score(data.X_train, data.y_train)
    #         test_score = model.score(data.X_test, data.y_test)
    #         logger.log("train_score", train_score)
    #         logger.log("loss", model.loss(data.X_train, data.y_train))
    #         logger.log("weights", np.mean(model.weights))
    #         logger.log("bias", np.mean(model.bias))
    #         logger.log("epoch", i * log_interval)
    #         logger.next_step()

    # predictions = model.predict(data.X_test)
    # proba = model.predict_proba(data.X_test)
    # predictions, predict_time = experiment.predict(data, model)
    # ground_truth = data.y_test
    # time_dict = {"fit_time": fit_time, "predict_time": predict_time}
    # score_dict = experiment.score(data, predictions)
    # output_files = experiment.save(
    #     **experiment.files,
    #     data=data,
    #     model=model,
    #     ground_truth=ground_truth,
    #     predictions=predictions,
    #     time_dict=time_dict,
    #     score_dict=score_dict,
    #     params=params,
    # )
    # for file in output_files:
    #     assert file.exists(), f"File {file} does not exist."
    # plots = params.pop("plots")
    # yb_model = classifier(model)
    # path = files.pop("path")
    # for name in classification_visualizers.keys():
    #     visualizer = classification_visualizers[name]
    #     if len(set(data.y_train)) > 2:
    #         viz = visualizer(
    #             yb_model,
    #             X_train=data.X_train,
    #             y_train=data.y_train,
    #             classes=[int(y) for y in np.unique(data.y_train)],
    #         )
    #     elif len(set(data.y_train)) == 2:
    #         viz = visualizer(
    #             yb_model,
    #             X_train=data.X_train,
    #             y_train=data.y_train,
    #             binary = True
    #         )
    #     viz.show(outpath=Path(path, name))
    #     assert Path(path, str(name)+".png").is_file(), f"File {name} does not exist."
    #     plt.gcf().clear()
