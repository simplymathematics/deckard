import json
import logging
import os
import pickle
from pathlib import Path
from time import process_time
from typing import List
import numpy as np
import pandas as pd
import yaml
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification.scikitlearn import ScikitlearnSVC
from art.utils import to_categorical
from pandas import DataFrame
from sklearn.svm import SVC
from tqdm import tqdm
from hydra.utils import instantiate


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def parse_folder(
    folder,
    exclude=[
        "probabilities",
        "predictions",
        "plots",
        "ground_truth",
        "attack_predictions",
        "attack_probabilities",
        "samples",
    ],
) -> pd.DataFrame:
    """
    Parse a folder containing json files and return a dataframe with the results, excluding the files in the exclude list.
    :param folder: Path to folder containing json files
    :param exclude: List of files to exclude. Default: ['probabilities', 'predictions', 'plots', 'ground_truth'].
    :return: Pandas dataframe with the results
    """
    folder = Path(folder)
    results = {}
    results[folder] = {}
    logger.debug(f"Parsing folder {folder}...")
    for file in folder.glob("*.json"):
        if Path(file).stem in exclude:
            continue
        else:
            with open(file, "r") as f:
                results[folder][Path(file).stem] = json.load(f)
    return pd.DataFrame(results).T


def flatten_results(results):
    """
    Flatten a dataframe containing json files. So that each json dict entry becomes a column with dot notation (e.g. "key1.subkey1")
    :param results: Pandas dataframe containing json files
    """
    new_results = pd.DataFrame()
    logger.debug("Flattening results...")
    for col in results.columns:
        tmp = pd.json_normalize(results[col])
        new_results = pd.concat([new_results, tmp], axis=1)
    return new_results


def parse_results(result_dir, flatten=True):
    """
    Recursively parse a directory containing json files and return a dataframe with the results.
    :param result_dir: Path to directory containing json files
    :param regex: Regex to match folders to parse. Default: "*/*"
    :param flatten: Whether to flatten the results. Default: True
    :return: Pandas dataframe with the results
    """
    result_dir = Path(result_dir)
    assert result_dir.is_dir(), f"Result directory {result_dir} does not exist."
    results = pd.DataFrame()
    logger.debug("Parsing results...")
    total = len(list(Path(result_dir).iterdir()))
    logger.info(f"Parsing {total} folders...")
    for folder in Path(result_dir).iterdir():
        tmp = parse_folder(folder)
        if flatten is True:
            tmp = flatten_results(tmp)
        tmp = tmp.loc[:, ~tmp.columns.duplicated()]
        results = pd.concat([results, tmp])
    return results


def set_for_keys(my_dict, key_arr, val) -> dict:
    """
    Set val at path in my_dict defined by the string (or serializable object) array key_arr.
    :param my_dict: Dictionary to set value in
    :param key_arr: Array of keys to set value at
    :param val: Value to set
    :return: Dictionary with value set
    """
    current = my_dict
    for i in range(len(key_arr)):
        key = key_arr[i]
        if key not in current:
            if i == len(key_arr) - 1:
                current[key] = val
            else:
                current[key] = {}
        else:
            if type(current[key]) is not dict:
                logger.info(
                    "Given dictionary is not compatible with key structure requested",
                )
                raise ValueError("Dictionary key already occupied")
        current = current[key]
    return my_dict


def unflatten_results(df, sep=".") -> List[dict]:
    """
    Unflatten a dataframe with dot notation columns (e.g. "key1.subkey1") into a list of dictionaries.
    :param df: Pandas dataframe with dot notation columns
    :param sep: Separator to use. Default: "."
    :return: List of dictionaries
    """
    logger.debug("Unflattening results...")
    result = []
    for _, row in df.iterrows():
        parsed_row = {}
        for idx, val in row.iteritems():
            if val == val:
                keys = idx.split(sep)
                parsed_row = set_for_keys(parsed_row, keys, val)
        result.append(parsed_row)
    return result


def retrain_loop(
    clf, X_train, y_train, X_test, y_test, atk, attack_size, epochs,
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
                i, ben_time, ben_score, adv_time, adv_score,
            ),
        )
        logger.info(
            "Epoch: {} - Benign Time: {} - Benign Score: {} - Adversarial Time: {} - Adversarial Score: {}".format(
                i, ben_time, ben_score, adv_time, adv_score,
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
results = pd.read_csv("output/train.csv")
# Some convenient variable names
# input_size = results["data.generate.kwargs.n_samples"] * results["data.generate.kwargs.n_features"]
results["Kernel"] = results["model.init.kwargs.kernel"].copy()
# results["Features"] = results["data.generate.kwargs.n_features"].copy()
# results["Samples"] = results["data.sample.train_size"].copy()
# results["input_size"] = input_size
# Clean up results
if "Unnamed: 0" in results.columns:
    del results["Unnamed: 0"]
for col in results.columns:
    if col == "data.name" and isinstance(results[col][0], list):
        results[col] = results[col].apply(lambda x: x[0])
# Subset results
# subset = results[results["data.sample.train_size"] == 10000]
# subset = subset[subset["data.generate.kwargs.n_features"] == 100]
with open("conf/model/best_rbf.yaml", "r") as f:
    best_rbf = yaml.safe_load(f)
best_rbf["init"].pop("_target_", None)
best_rbf["init"].pop("name", None)
with open("conf/model/best_poly.yaml", "r") as f:
    best_poly = yaml.safe_load(f)
best_poly["init"].pop("_target_", None)
best_poly["init"].pop("name", None)
with open("conf/model/best_linear.yaml", "r") as f:
    best_lin = yaml.safe_load(f)
best_lin["init"].pop("_target_", None)
best_lin["init"].pop("name", None)
rbf_model = SVC(**best_rbf["init"])
# Poly
poly_model = SVC(**best_poly["init"])
# Linear
linear_model = SVC(**best_lin["init"])
# Load Data
with open("conf/data/attack.yaml", "r") as f:
    data = yaml.safe_load(f)
data = instantiate(data)
X_train, X_test, y_train, y_test = data()
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
epochs = 20
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
    confidence_df = pd.DataFrame()
    for folder in tqdm(os.listdir("output/reports/attack")):
        confidence_ser = pd.Series()
        if Path("output/reports/attack", folder, "adv_probabilities.json").exists():
            with open(
                Path("output/reports/attack", folder, "adv_probabilities.json"), "r",
            ) as f:
                probs = json.load(f)
            probs = np.array(probs)
            false_confidence = y_test[: len(probs)] - probs[:, 1]
            avg_prob = np.mean(false_confidence)
            with open(
                Path("output/reports/attack", folder, "score_dict.json"), "r",
            ) as f:
                try:
                    scores = json.load(f)
                except:  # noqa E722
                    scores = {}
            if "False Confidence" in scores:
                del scores["False Confidence"]
            scores[f"False Confidence before retraining {name.capitalize()}"] = avg_prob
            with open(
                Path("output/reports/attack", folder, "score_dict.json"), "w",
            ) as f:
                json.dump(scores, f)
            yaml_file = Path("output/reports/attack", folder, "params.yaml")
            json_file = Path("output/reports/attack", folder, "params.json")
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
            attack_params = params["attack"]["init"]["kwargs"]
            attack_params.update({"name": params["attack"]["init"]["name"]})
            confidence_ser["Kernel"] = name
            confidence_ser["Average False Confidence"] = avg_prob
            # print(f"Shape of confidence ser: {confidence_ser.shape}")
            attack_ser = pd.Series(attack_params)
            confidence_ser = confidence_ser._append(attack_ser)
            # print(f"Shape of confidence ser: {confidence_ser.shape}")
            if "Unnamed: 0" in confidence_df.columns:
                del confidence_df["Unnamed: 0"]
            confidence_df = confidence_df._append(confidence_ser, ignore_index=True)
            # print(f"Shape of confidence df: {confidence_df.shape}")
            # print(confidence_df.head())
            # input("Press Enter to continue...")
        else:
            pass
    df1 = pd.concat([df1, confidence_df], ignore_index=True)
    Path("plots").mkdir(parents=True, exist_ok=True)
    df1.to_csv(Path("plots", "before_retrain_confidence.csv"))
    # Train model on attack samples
    print(f"Training Model {i} of {len(models)}")
    if not Path("retrain", name, f"{epochs}.pkl").exists():
        results, outputs = retrain_loop(
            model, X_train, y_train, X_test, y_test, atk, epochs=epochs, attack_size=50,
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
    for folder in tqdm(os.listdir("output/reports/attack")):
        confidence_ser = pd.Series()
        if Path("output/reports/attack", folder, "adv_probabilities.json").exists():
            with open(
                Path("output/reports/attack", folder, "adv_probabilities.json"), "r",
            ) as f:
                probs = json.load(f)
            probs = np.array(probs)
            false_confidence = y_test[: len(probs)] - probs[:, 1]
            avg_prob = np.mean(false_confidence)
            pd.DataFrame(probs).to_csv(
                Path(
                    "output/reports/attack",
                    folder,
                    f"probs_after_retraining_{name}.csv",
                ),
            )
            with open(
                Path("output/reports/attack", folder, "score_dict.json"), "r",
            ) as f:
                scores = json.load(f)
            if "False Confidence" in scores:
                del scores["False Confidence"]
            scores[f"False Confidence {name.capitalize()}"] = avg_prob
            with open(
                Path("output/reports/attack", folder, "score_dict.json"), "w",
            ) as f:
                json.dump(scores, f)
            if Path("output/reports/attack", folder, "params.yaml").exists():
                with open(
                    Path("output/reports/attack", folder, "params.yaml"), "r",
                ) as f:
                    params = yaml.safe_load(f)
            elif Path("output/reports/attack", folder, "params.json").exists():
                with open(
                    Path("output/reports/attack", folder, "params.json"), "r",
                ) as f:
                    params = json.load(f)
            else:
                logger.warning(f"No params file found for {folder}")
                continue
            attack_params = params["attack"]["init"]["kwargs"]
            attack_params.update({"name": params["attack"]["init"]["name"]})
            confidence_ser["Kernel"] = name
            confidence_ser["Average False Confidence After Retraining"] = avg_prob
            attack_ser = pd.Series(attack_params)
            confidence_ser = confidence_ser._append(attack_ser)
            if "Unnamed: 0" in confidence_df.columns:
                del confidence_df["Unnamed: 0"]
            confidence_df = confidence_df._append(confidence_ser, ignore_index=True)
    df2 = pd.concat([df2, confidence_df], ignore_index=True)
    df2.to_csv(Path("plots", "after_retrain_confidence.csv"))
    i += 1
