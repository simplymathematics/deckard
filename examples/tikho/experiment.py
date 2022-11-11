

import collections
import numpy as np
from pathlib import Path
import yaml
from time import process_time
from data import Data
from model import Model

from tqdm import tqdm

from json_mixin import JSONMixin
from yellowbrick.target import ClassBalance, BalancedBinningReference, FeatureCorrelation
from yellowbrick.features import Rank2D, RadViz, PCA, Manifold, Rank1D, ParallelCoordinates
target_visualizers ={
    "bins" : BalancedBinningReference,
    "class_balance" : ClassBalance,
    # "correlation" : FeatureCorrelation,
}

feature_visualizers = {
    "radviz" : RadViz,
    "pca" : PCA,
    "manifold" : Manifold,
    "rank1d" : Rank1D,
    "rank2d" : Rank2D,
    "parallel" : ParallelCoordinates,
}




class Experiment(collections.namedtuple(typename = 'Experiment', field_names = 'data, model, is_fitted, scorers, plots, files, fit', defaults = ({},{}, {}, {})), JSONMixin):    
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    def fit(self, data:dict, model:object) -> tuple:
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
        return model, result/len(data.X_train)

    def predict(self, data:dict, model:object) -> tuple:
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
        return predictions, result/len(data.X_test)

    def load(config: str) -> tuple:
        """
        Loads data, model from the config file.
        :param config: str, path to config file.
        :returns: tuple(dict, object), (data, model).
        """
        yaml.add_constructor('!Data:', Data)
        yaml.add_constructor('!Model:', Model)
        data_document = """!Data:\n""" + str(dict(experiment.data))
        model_document = """!Model:\n""" + str(dict(experiment.model))
        data = yaml.load(data_document, Loader = yaml.Loader)
        data = data.load()
        model = yaml.load(model_document, Loader = yaml.Loader)
        model = model.load()
        return data, model
    






from dvclive import Live
from dvc.api import params_show
params = params_show("params.yaml")
files = params['files']
epoch = params['fit']['epochs']
log_interval = params['fit']['log_interval']
learning_rate = params['fit']['learning_rate']
yaml.add_constructor('!Experiment:', Experiment)
experiment = yaml.load( "!Experiment:\n" + str(params), Loader = yaml.Loader)
data, model = Experiment.load(experiment)
logger = Live(path = Path(files['path']), report = "html")
epochs = round(int(epoch/log_interval))
for i in tqdm(range(epochs)):
    start = process_time()
    clf = model.fit(data.X_train, data.y_train, learning_rate = learning_rate)
    if i % log_interval == 0:
        fit_time = process_time() - start
        logger.log("time", fit_time)
        train_score = clf.score(data.X_train, data.y_train)
        test_score = clf.score(data.X_test, data.y_test)
        logger.log("train_score", train_score)
        logger.log("loss", clf.loss(data.X_train, data.y_train))
        logger.log("weights", np.mean(clf.weights))
        logger.log("bias", np.mean(clf.bias))
        logger.log("epoch", i * log_interval)
        logger.next_step() 
predictions = clf.predict(data.X_test)
proba = clf.predict_proba(data.X_test)
predictions, predict_time = experiment.predict(data, model)
ground_truth = data.y_test
time_dict = {"fit_time": fit_time, "predict_time": predict_time}
score_dict = experiment.score(data, predictions)
print(score_dict)
files = experiment.save(**experiment.files, data = data, model = model, ground_truth = ground_truth, predictions = predictions, time_dict = time_dict, score_dict = score_dict)
for file in files:
    assert file.exists(), f"File {file} does not exist."
