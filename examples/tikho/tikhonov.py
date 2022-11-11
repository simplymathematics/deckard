
import argparse
import pickle
from pathlib import Path
from time import process_time
import matplotlib.pyplot as plt
import numpy as np
import yaml
from dvclive import Live
from dvc.api import params_show
from pandas import DataFrame, Series
from tqdm import tqdm
from yellowbrick.classifier import classification_report, confusion_matrix
from yellowbrick.contrib.wrapper import classifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data import Data
from model import Model
class TikhonovClassifier:
    def __init__(self,  mtype:str = "linear", scale:float = 0.0):
        
        self.type = mtype
        self.scale = scale

        if mtype.lower() == "linear":
            self.loss = self.lin_loss
            self.gradient = self.lin_gradient
        elif mtype.lower() == "logistic":
            self.loss = self.log_loss
            self.gradient = self.log_gradient
        else:
            raise NotImplementedError("Model type not implemented")

    def sigmoid(self, z):
        return np.divide(1, 1 + np.exp(-z))

    def lin_loss(self, x, y):
        y_hat = x @ self.weights + self.bias
        errors = y_hat - y
        squared = 0.5 * np.linalg.norm(errors) ** 2
        dydx = self.weights
        tikho = 0.5 * np.sum(dydx**2)
        return squared + tikho * self.scale

    def lin_gradient(self, x, y):
        # || x @ weights + bias - y || ^2
        reg = 2 * self.weights * self.scale
        gradL_w = 2 * (x @ self.weights + self.bias - y) @ x + reg
        gradL_b = 2 * (x @ self.weights + self.bias - y)
        return (gradL_w, gradL_b)
    
    def log_loss(self, x, y):
        y_hat = self.sigmoid(x @ self.weights + self.bias)
        errors = np.mean(y * (np.log(y_hat)) + (1 - y) * (1 - y_hat))
        # + scale/2 * np.mean(weights ** 2)
        return errors + self.scale / 2 * np.mean(self.weights**2)

    def log_gradient(self, x, y):
        y_hat = self.sigmoid(x @ self.weights + self.bias)
        reg = 2 * self.weights * self.scale
        gradL_w = 2 * (x @ self.weights + self.bias - y) @ x + 2 * reg
        gradL_b = np.mean(y_hat - y)
        return (gradL_w, gradL_b)

    def fit(self, X_train, y_train, learning_rate=1e-6, epochs = 1000):
        self.weights = np.ones((X_train.shape[1])) * 1e-8 if not hasattr(self, "weights") else self.weights
        self.bias = 0.0 if not hasattr(self, "bias") else self.bias
        for i in range(epochs):
            L_w, L_b = self.gradient(X_train, y_train)
            self.weights -= L_w * learning_rate
            self.bias -= L_b * learning_rate
        return self

    def predict(self, x):
        # print(x.shape, self.weights.shape, self.bias.shape)
        x_dot_weights = x @ self.weights.T
        return [1 if p > 0.5 else 0 for p in x_dot_weights]

    def predict_proba(self, x):
        x_dot_weights = x @ self.weights.T
        return x_dot_weights

    def score(self, x, y):
        # print(x.shape, weights.shape, bias.shape)
        x_dot_weights = x @ self.weights.T
        y_test = [1 if p > 0.5 else 0 for p in x_dot_weights]
        return np.mean(y == y_test)

if __name__ == "__main__":
    params = params_show()
    if "plots" in params:
        plots = params.pop('plots')
    if "scorers" in params:
        metrics = params.pop('scorers')
    if "files" in params:
        files = params.pop('files')
    if "model" in params:
        model = params.pop('model')
        yaml.add_constructor('!Model:', Model)
        model = yaml.load("!Model:\n" + str(model), Loader=yaml.FullLoader)
    else:
        raise ValueError("No model specified in params.yaml")
    if "data" in params:
        data = params.pop("data")
        yaml.add_constructor('!Data:', Data)
        data = yaml.load("!Data:\n" + str(data), Loader=yaml.FullLoader)
        data = data.load()
        X_train = data.X_train
        y_train = data.y_train
        X_test = data.X_test
        y_test = data.y_test
    else:
        raise ValueError("No data specified in params.yaml")
    
    learning_rate = params.pop("learning_rate")
    log_interval = params.pop("log_interval")
    epochs = int(round(params.pop("epochs")/log_interval))
    logger = Live(path = Path(plots['path']), report = "html")
    
    for i in tqdm(range(epochs)):
        start = process_time()
        model = model.fit(X_train, y_train, learning_rate = learning_rate)
        if i % log_interval == 0:
            logger.log("time", process_time() - start)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            logger.log("train_score", train_score)
            logger.log("loss", model.loss(X_train, y_train))
            logger.log("weights", np.mean(model.weights))
            logger.log("bias", np.mean(model.bias))
            logger.log("epoch", i * log_interval)
            logger.next_step() 
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)
    ground_truth = y_test
    df = DataFrame({
        "predictions": predictions, 
        "probabilities" : proba, 
        "ground_truth": ground_truth
        })
    
    df.to_csv(Path(files['path'], files['predictions']), index=False)
    acc = accuracy_score(ground_truth, predictions)
    prec = precision_score(ground_truth, predictions)
    rec = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    ser = Series({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
    ser.to_json(Path(files['path'], metrics['scores']))
    yb_model = classifier(model)
    path = plots.pop("path")
    i = 0
    for visualizer in [classification_report, confusion_matrix]:
        name = list(plots.values())[i]
        viz = visualizer(yb_model, X_train = X_train, y_train = y_train, classes=[int(y) for y in np.unique(y_test)])
        viz.show(outpath=Path(path, name))
        plt.gcf().clear()
        i += 1
    model = Path(files['path'], files.pop("model"))
    model.parent.mkdir(parents=True, exist_ok=True)
    with open(model, "wb") as f:
        pickle.dump(model, f)
    f"Unused Params: {params}"
