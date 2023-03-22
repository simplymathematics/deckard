import json
import logging
import os
import pickle
from pathlib import Path
from time import process_time

import numpy as np
import pandas as pd
import yaml
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification.scikitlearn import ScikitlearnSVC
from art.utils import to_categorical
from pandas import DataFrame
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

from deckard.layers.compile import parse_results, unflatten_results  # flatten_results,

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def retrain_loop(
    clf,
    X_train,
    y_train,
    X_test,
    y_test,
    atk,
    attack_size,
    epochs,
) -> tuple:
    i = 0
    results = []
    for epochs in tqdm(range(epochs), desc="Epochs"):
        # Benign Experiment
        logger.info("Epoch: {} - Benign Training".format(i))
        try:
            y_train = to_categorical(y_train)
        except:  # noqa E722
            pass
        try:
            y_test = to_categorical(y_test)
        except:  # noqa E722
            pass
        start = process_time()
        clf.fit(X_train, y_train)
        ben_time = (process_time() - start) / len(X_train)
        ben_predictions = clf.predict(X_test)
        ben_score = np.mean(
            np.argmax(ben_predictions, axis=1) == np.argmax(y_test, axis=1),
        )
        ben_loss = np.mean(ben_predictions[:, 0] - y_test[:, 0])
        # Adversarial Experiment
        logger.info("Epoch: {} - Adversarial Training".format(i))
        start = process_time()
        adv = atk.generate(X_test[:attack_size])
        adv_time = (process_time() - start) / attack_size
        adv_predictions = clf.predict(adv)
        adv_score = np.mean(
            np.argmax(adv_predictions, axis=1)
            == np.argmax(y_test[:attack_size], axis=1),
        )
        adv_loss = np.mean(adv_predictions[:, 0] - y_test[:attack_size, 0])
        # Append Adversarial Examples to Training Set
        X_train = np.concatenate((X_train, adv), axis=0)
        adv_labels = to_categorical(y_test[:attack_size, 0])
        y_train = np.concatenate((y_train, adv_labels), axis=0)
        i += 1
        # Save Results
        results.append(
            {
                "ben_time": ben_time,
                "ben_score": ben_score,
                "adv_time": adv_time,
                "adv_score": adv_score,
                "ben_loss": ben_loss,
                "adv_loss": adv_loss,
                "attack_size": attack_size,
                "train_size": len(X_train),
                "test_size": attack_size,
            },
        )
        outputs = {
            "ben_predictions": DataFrame(ben_predictions),
            "adv_predictions": DataFrame(adv_predictions),
        }
        # Some Logging
        print(
            "Epoch: {} - Benign Time: {} - Benign Score: {} - Adversarial Time: {} - Adversarial Score: {}".format(
                i,
                ben_time,
                ben_score,
                adv_time,
                adv_score,
            ),
        )
        logger.info(
            "Epoch: {} - Benign Time: {} - Benign Score: {} - Adversarial Time: {} - Adversarial Score: {}".format(
                i,
                ben_time,
                ben_score,
                adv_time,
                adv_score,
            ),
        )
    results = pd.DataFrame(results)
    return results, outputs


def save_results_and_outputs(results, outputs, path="retrain") -> list:
    Path(path).mkdir(parents=True, exist_ok=True)
    for output in outputs:
        pd.DataFrame(outputs[output]).to_csv(f"{path}/{output}.csv")
        assert Path(f"{path}/{output}.csv").exists(), f"Problem saving {path}/{output}"
    pd.DataFrame(results).to_csv(f"{path}/results.csv")
    assert Path(
        f"{path}/results.csv",
    ).exists(), f"Problem saving results to {path}/results.csv"


# Parse Model Results
if not Path("reports/model_queue/results.csv").exists():
    results = parse_results("reports/model_queue/")
    results.to_csv("reports/model_queue/results.csv")
else:
    results = pd.read_csv("reports/model_queue/results.csv")
# Some convenient variable names
input_size = results["data.generate.n_samples"] * results["data.generate.n_features"]
results["Kernel"] = results["model.init.kernel"].copy()
results["Features"] = results["data.generate.n_features"].copy()
results["Samples"] = results["data.sample.train_size"].copy()
results["input_size"] = input_size
# Clean up results
if "Unnamed: 0" in results.columns:
    del results["Unnamed: 0"]
for col in results.columns:
    if col == "data.name" and isinstance(results[col][0], list):
        results[col] = results[col].apply(lambda x: x[0])
results = results[results["model.init.kernel"] != "sigmoid"]
# Save results
results.to_csv("plots/model_results.csv")
# Subset results
subset = results[results["data.sample.train_size"] == 10000]
subset = subset[subset["data.generate.n_features"] == 100]
# Generate Models
best_rbf = (
    subset[subset["model.init.kernel"] == "rbf"]
    .sort_values(by="accuracy", ascending=False)
    .head(1)
)
best_poly = (
    subset[subset["model.init.kernel"] == "poly"]
    .sort_values(by="accuracy", ascending=False)
    .head(1)
)
best_lin = (
    subset[subset["model.init.kernel"] == "linear"]
    .sort_values(by="accuracy", ascending=False)
    .head(1)
)
# RBF
rbf_unflattened = unflatten_results(best_rbf)
rbf_unflattened[0]["model"]["init"].pop("name")
rbf_model = SVC(**rbf_unflattened[0]["model"]["init"])
# Poly
poly_unflattened = unflatten_results(best_poly)
poly_unflattened[0]["model"]["init"].pop("name")
poly_model = SVC(**poly_unflattened[0]["model"]["init"])
# Linear
linear_unflattened = unflatten_results(best_lin)
linear_unflattened[0]["model"]["init"].pop("name")
linear_model = SVC(**linear_unflattened[0]["model"]["init"])
# Generate Data
data_generate = rbf_unflattened[0]["data"]["generate"]
data_init = rbf_unflattened[0]["data"]["sample"]
X, y = make_classification(**data_generate)
data_init.pop("time_series")
data_init["stratify"] = y
X_train, X_test, y_train, y_test = train_test_split(X, y, **data_init)
# Fit Models
rbf_model.fit(X_train, y_train)
poly_model.fit(X_train, y_train)
linear_model.fit(X_train, y_train)
models = [rbf_model, poly_model, linear_model]
model_names = ["rbf", "poly", "linear"]
# ART Models
art_models = [
    ScikitlearnSVC(model=rbf_model),
    ScikitlearnSVC(model=poly_model),
    ScikitlearnSVC(model=linear_model),
]
i = 1
epochs = 1

df1 = pd.DataFrame()
df2 = pd.DataFrame()
for model in art_models:
    # Define model name
    name = model_names[i - 1]
    # Define attack
    atk = ProjectedGradientDescent(
        estimator=model,
        eps=1,
        eps_step=0.1,
        max_iter=10,
        targeted=False,
        num_random_init=0,
        batch_size=10,
    )
    # Retrain model if not already done
    if not Path("best_models", f"{name}.pickle").exists():
        raise ValueError(
            f"Model best_models/{name}.pickle not found. Please run the model queue first.",
        )
    else:  # Otherwise load model
        with open(Path("best_models", f"{name}.pickle"), "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, ScikitlearnSVC):
            model = ScikitlearnSVC(model=model)
    # Get false confidence scores
    confidence_df = pd.DataFrame()
    for folder in tqdm(os.listdir("reports/attack")):
        confidence_ser = pd.Series()
        if Path("reports/attack", folder, "samples.json").exists():
            with open(Path("reports/attack", folder, "samples.json"), "r") as f:
                samples = json.load(f)
            samples = pd.DataFrame(samples).head().to_numpy()
            probs = model.model.predict_proba(samples)
            false_confidence = y_test[: len(probs)] - probs[:, 1]
            avg_prob = np.mean(false_confidence)
            with open(Path("reports/attack", folder, "scores.json"), "r") as f:
                try:
                    scores = json.load(f)
                except:  # noqa E722
                    scores = {}
            if "False Confidence" in scores:
                del scores["False Confidence"]
            scores[f"False Confidence before retraining {name.capitalize()}"] = avg_prob
            with open(Path("reports/attack", folder, "scores.json"), "w") as f:
                json.dump(scores, f)
            yaml_file = Path("reports/attack", folder, "params.yaml")
            json_file = Path("reports/attack", folder, "params.json")
            if yaml_file.exists():
                params_file = yaml_file
                with open(params_file, "r") as f:
                    params = yaml.safe_load(f)
            elif json_file.exists():
                params_file = json_file
                with open(params_file, "r") as f:
                    params = json.load(f)
            else:
                raise ValueError(f"No params file found for {folder}")
            attack_params = params["attack"]["init"]
            confidence_ser["Kernel"] = name
            confidence_ser["Average False Confidence"] = avg_prob
            attack_ser = pd.Series(attack_params)
            confidence_ser = confidence_ser.append(attack_ser)
            if "Unnamed: 0" in confidence_df.columns:
                del confidence_df["Unnamed: 0"]
            confidence_df = confidence_df.append(confidence_ser, ignore_index=True)
        else:
            pass
    df1 = df1.append(confidence_df, ignore_index=True)
    df1.to_csv(Path("plots", "before_retrain_confidence.csv"))
    # Train model on attack samples
    print(f"Training Model {i} of {len(models)}")
    if not Path("retrain", name, f"{epochs}.pkl").exists():
        results, outputs = retrain_loop(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            atk,
            epochs=epochs,
            attack_size=100,
        )
        save_results_and_outputs(results, outputs, path=f"retrain/{name}/")
        Path("retrain", name).mkdir(parents=True, exist_ok=True)
        with Path("retrain", name, f"{epochs}.pkl").open("wb") as f:
            pickle.dump(model, f)
    else:
        with open(Path("retrain", name, f"{epochs}.pkl"), "rb") as f:
            model = pickle.load(f)
    print(f"Evaluating Model {i} of {len(models)}")
    # Get false confidence scores after retraining
    confidence_df = pd.DataFrame()
    for folder in tqdm(os.listdir("reports/attack")):
        confidence_ser = pd.Series()
        if Path("reports/attack", folder, "samples.json").exists():
            with open(Path("reports/attack", folder, "samples.json"), "r") as f:
                samples = json.load(f)
            samples = pd.DataFrame(samples).head(100).to_numpy()
            probs = model.model.predict_proba(samples)
            false_confidence = y_test[: len(probs)] - probs[:, 1]
            avg_prob = np.mean(false_confidence)
            pd.DataFrame(probs).to_csv(
                Path("reports/attack", folder, f"probs_after_retraining_{name}.csv"),
            )
            with open(Path("reports/attack", folder, "scores.json"), "r") as f:
                scores = json.load(f)
            if "False Confidence" in scores:
                del scores["False Confidence"]
            scores[f"False Confidence {name.capitalize()}"] = avg_prob
            with open(Path("reports/attack", folder, "scores.json"), "w") as f:
                json.dump(scores, f)
            if Path("reports/attack", folder, "params.yaml").exists():
                with open(Path("reports/attack", folder, "params.yaml"), "r") as f:
                    params = yaml.safe_load(f)
            elif Path("reports/attack", folder, "params.json").exists():
                with open(Path("reports/attack", folder, "params.json"), "r") as f:
                    params = json.load(f)
            else:
                logger.warning(f"No params file found for {folder}")
                continue
            attack_params = params["attack"]["init"]
            confidence_ser["Kernel"] = name
            confidence_ser["Average False Confidence After Retraining"] = avg_prob
            attack_ser = pd.Series(attack_params)
            confidence_ser = confidence_ser.append(attack_ser)
            if "Unnamed: 0" in confidence_df.columns:
                del confidence_df["Unnamed: 0"]
            confidence_df = confidence_df.append(confidence_ser, ignore_index=True)
    df2 = df2.append(confidence_df, ignore_index=True)
    df2.to_csv(Path("plots", "after_retrain_confidence.csv"))
    i += 1
