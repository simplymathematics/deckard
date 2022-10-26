
import argparse
import logging
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from dvclive import Live
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from dvclive import Live


from pathlib import Path

from tikhonov import TikhonovClassifier
import dvc.api

import dvc.api
from tikhonov import REPORT_FOLDER
def run_experiment(
    id_,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=1000,
    learning_rate=1e-6,
    input_noise = 0,
    output_noise = 0,
    scale = 0,
    **kwargs,
):  
    model.fit(X_train, y_train, learning_rate=learning_rate, epochs=epochs)
    predictions = model.predict(X_test)
    actual = y_test
    probabilities = model.predict_proba(X_test)
    id_ = id_
    df = pd.DataFrame(
        {"predictions": predictions, "actual": actual, "probabilities": probabilities}
    )
    try:
        f1 = f1_score(actual, predictions)
    except:
        f1 = None
    try:
        auc = roc_auc_score(actual, probabilities)
    except:
        auc = None
    try:
        accuracy = accuracy_score(actual, predictions)
    except:
        accuracy = None
    try:
        precision = precision_score(actual, predictions)
    except:
        precision = None
    try:
        recall = recall_score(actual, predictions)
    except:
        recall = None
    loss = model.loss(X_train, y_train)
    test_loss = np.mean(y_test - model.predict_proba(X_test))
    series = pd.Series(
        {
            "f1": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "scale": scale,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "training_loss": loss,
            "testing_loss": test_loss,
            "input_noise": input_noise,
            "output_noise": output_noise,
            "type":model.type,
        },
        name=id_,
    )

    met_file = "metrics.json"
    pred_file =  "predictions.json"
    Path(REPORT_FOLDER, id_).mkdir(exist_ok = True, parents = True)
    met_file = Path(REPORT_FOLDER, id_, met_file)
    pred_file = Path(REPORT_FOLDER, id_, pred_file)
    series.to_json(met_file, orient="index")
    df.to_json(pred_file, orient="records")

def get_data(output_noise=0, input_noise=0):
    X, y = make_classification(n_samples=10000, n_features=20, random_state=42, n_classes=2, n_informative=4, scale = 10, n_redundant=0, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if input_noise > 0:
        X_train = X_train + np.random.normal(0, input_noise, X_train.shape)
    elif input_noise < 0:
        raise ValueError("Input noise must be positive")
    if output_noise > 0:
        X_test = X_test + np.random.normal(0, output_noise, X_test.shape)
    elif output_noise < 0:
        raise ValueError("Output noise must be positive")
    return X_train, X_test, y_train, y_test


def dohmatob(X_train, probability, error, mink=2):
    ds = []
    for column in range(X_train.shape[1]):
        std = np.std(X_train[:, column])
        ds.append(std * np.sqrt(2 * np.log(1 / error) / probability))
    return np.linalg.norm(ds, mink=mink)

logger = logging.getLogger(__name__)
if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--mink", type=int, default=2)
    parser.add_argument("--input_noise", type=float, default=0.0)
    parser.add_argument("--output_noise", type=float, default=0.0)
    parser.add_argument("--type", type=str, default="linear")
    parser.add_argument("--queue", type=str, default="queue.csv")
    args = parser.parse_args()

    from tqdm import tqdm
    from random import randint
    failures = []
    queue = pd.read_csv("queue.csv", index_col = None)
    queue.sample(frac = 1, random_state = randint(0, 2**32) )
    for i in tqdm(range(len(queue)),  "Big Grid Search"):
        try:
            entry = queue.iloc[0]
        except IndexError:
            logger.info("Queue is empty")
            break
        id_ = str(uuid4())
        name = str(entry)
        path = REPORT_FOLDER
        full_path = Path(path, id_)
        full_path.mkdir(exist_ok=True, parents=True)
        params_path = Path(full_path, "params.json")
        entry.to_csv(params_path)
        print("Running Experiment: {}".format(entry))
        live = Live(path = Path(full_path, id_), report = "html")
        X_train, X_test, y_train, y_test = get_data(
            entry["output_noise"], entry["input_noise"]
        )
        if entry['type'] == "linear":
            model = TikhonovClassifier(scale = entry['scale'], input_noise = entry['input_noise'], output_noise = entry['output_noise'], mtype = "linear", logger = live)
        elif entry['type'] == "logistic":
            model = TikhonovClassifier(scale = entry['scale'], input_noise = entry['input_noise'], output_noise = entry['output_noise'], mtype = "logistic", logger = live)
        else:
            raise ValueError("Type must be linear or logistic")
        del entry["type"]
        run_experiment(id_, model, X_train, y_train, X_test, y_test, output_noise =entry['output_noise'], input_noise = entry['input_noise'], epochs = entry['epochs'], learning_rate = entry['learning_rate'], scale =entry['scale'],)
        queue.drop(index=queue.index[0], axis=0, inplace=True)
        queue.to_csv("queue.csv")
        pd.DataFrame(failures).to_csv("failures.txt")

    # print(dvc.api.params_show('params.yaml'))
