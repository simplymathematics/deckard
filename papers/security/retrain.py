import json
import logging
import pickle
from pathlib import Path
from time import process_time

import numpy as np
import pandas as pd
import yaml
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification.scikitlearn import ScikitlearnSVC
from art.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SECURITY_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SECURITY_DIR / "config"
OUTPUTS_DIR = SECURITY_DIR / "outputs"
PLOTS_DIR = SECURITY_DIR / "plots"
RETRAIN_DIR = SECURITY_DIR / "retrain"


def _load_model_config(kernel: str) -> dict:
    cfg_file = CONFIG_DIR / "model" / f"{kernel}.yaml"
    with cfg_file.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("model_params", {})


def _load_data_from_security_config() -> tuple:
    data_file = CONFIG_DIR / "data" / "classification.yaml"
    with data_file.open("r") as f:
        data_cfg = yaml.safe_load(f) or {}

    params = data_cfg.get("data_params", {})
    n_samples = int(params.get("n_samples", 10000))
    n_features = int(params.get("n_features", 100))
    n_classes = int(params.get("n_classes", 2))
    random_state = int(params.get("random_state", 0))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
    )

    train_size = int(data_cfg.get("train_size", 10000))
    test_size = int(data_cfg.get("test_size", 1000))
    train_ratio = train_size / float(train_size + test_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_ratio,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def retrain_loop(
    clf, X_train, y_train, X_test, y_test, atk, attack_size, epochs
):
    i = 0
    results = []
    for _ in tqdm(range(epochs), desc="Epochs"):
        logger.info("Epoch: %s - Benign Training", i)
        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)

        start = process_time()
        clf.fit(X_train, y_train_cat)
        ben_time = (process_time() - start) / len(X_train)
        ben_predictions = clf.predict(X_test)
        ben_score = np.mean(
            np.argmax(ben_predictions, axis=1) == np.argmax(y_test_cat, axis=1),
        )
        ben_loss = np.mean(ben_predictions[:, 0] - y_test_cat[:, 0])

        logger.info("Epoch: %s - Adversarial Training", i)
        start = process_time()
        adv = atk.generate(X_test[:attack_size])
        adv_time = (process_time() - start) / attack_size
        adv_predictions = clf.predict(adv)
        adv_score = np.mean(
            np.argmax(adv_predictions, axis=1)
            == np.argmax(y_test_cat[:attack_size], axis=1),
        )
        adv_loss = np.mean(adv_predictions[:, 0] - y_test_cat[:attack_size, 0])

        X_train = np.concatenate((X_train, adv), axis=0)
        adv_labels = to_categorical(y_test_cat[:attack_size, 0])
        y_train_cat = np.concatenate((y_train_cat, adv_labels), axis=0)

        i += 1
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

    return pd.DataFrame(results)


def save_results(
    results: pd.DataFrame, kernel: str, epochs: int, model
) -> None:
    out_dir = RETRAIN_DIR / kernel
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    with (out_dir / f"{epochs}.pkl").open("wb") as f:
        pickle.dump(model, f)


def annotate_false_confidence() -> None:
    for trial_dir in (OUTPUTS_DIR / "logs").glob("*/*"):
        scores_file = trial_dir / "scores.json"
        if not scores_file.exists():
            continue
        with scores_file.open("r") as f:
            scores = json.load(f) or {}
        if "evasion_accuracy" in scores:
            scores["False Confidence"] = float(1.0 - scores["evasion_accuracy"])
            with scores_file.open("w") as f:
                json.dump(scores, f, indent=4)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = _load_data_from_security_config()

    kernels = ["rbf", "poly", "linear"]
    epochs = 20
    attack_size = 50

    for kernel in kernels:
        model_params = _load_model_config(kernel)
        base_model = SVC(**model_params)
        base_model.fit(X_train, y_train)
        art_model = ScikitlearnSVC(model=base_model)

        atk = ProjectedGradientDescent(
            estimator=art_model,
            eps=1,
            eps_step=0.1,
            max_iter=10,
            targeted=False,
            num_random_init=0,
            batch_size=10,
        )

        model_file = RETRAIN_DIR / kernel / f"{epochs}.pkl"
        if model_file.exists():
            with model_file.open("rb") as f:
                art_model = pickle.load(f)
            logger.info("Loaded existing retrained model for kernel=%s", kernel)
            continue

        logger.info("Retraining kernel=%s", kernel)
        results = retrain_loop(
            art_model,
            X_train,
            y_train,
            X_test,
            y_test,
            atk,
            attack_size=attack_size,
            epochs=epochs,
        )
        save_results(results, kernel, epochs, art_model)

    annotate_false_confidence()


if __name__ == "__main__":
    main()
