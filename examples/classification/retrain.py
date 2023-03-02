import pickle 
import json
import logging
from pathlib import Path
from time import process_time
import numpy as np
from tqdm import tqdm
from pandas import DataFrame
from sklearn.svm import SVC
from art.estimators.classification.scikitlearn import ScikitlearnSVC
from art.attacks.evasion import ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer
from art.utils import to_categorical

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def retrain_loop(clf, X_train, y_train, X_test, y_test, atk, attack_size, epochs) -> tuple:
    i = 0
    results = []
    for epochs in tqdm(range(epochs), desc="Epochs"):
        # Benign Experiment
        logger.info("Epoch: {} - Benign Training".format(i))
        start = process_time()
        clf.fit(X_train, y_train)
        ben_time = (process_time() - start)/len(X_train)
        ben_predictions = clf.predict(X_test)
        ben_score = np.mean(np.argmax(ben_predictions, axis=1) == np.argmax(y_test, axis=1))
        ben_loss = np.mean(ben_predictions[:,0] - y_test[:,0])
        # Adversarial Experiment
        logger.info("Epoch: {} - Adversarial Training".format(i))
        start = process_time()
        adv = atk.generate(X_test[:attack_size])
        adv_time = (process_time() - start)/attack_size
        adv_predictions = clf.predict(adv)
        adv_score = np.mean(np.argmax(adv_predictions, axis=1) == np.argmax(y_test[:attack_size], axis=1))
        adv_loss = np.mean(adv_predictions[:,0] - y_test[:attack_size,0])
        # Append Adversarial Examples to Training Set
        X_train = np.concatenate((X_train, adv), axis=0)
        adv_labels = to_categorical(np.argmax(adv_predictions, axis=1))
        y_train = np.concatenate((y_train, adv_labels), axis=0)
        i += 1
        # Save Results
        results.append({
            "ben_time": ben_time,
            "ben_score": ben_score,
            "adv_time": adv_time,
            "adv_score": adv_score,
            "ben_loss": ben_loss,
            "adv_loss": adv_loss,
            "attack_size": attack_size,
        })
        outputs = {
            "ben_predictions": DataFrame(ben_predictions).to_json(orient="records"),
            "adv_predictions": DataFrame(adv_predictions).to_json(orient="records"),
        }
        # Some Logging
        print("Epoch: {} - Benign Time: {} - Benign Score: {} - Adversarial Time: {} - Adversarial Score: {}".format(i, ben_time, ben_score, adv_time, adv_score))
        logger.info("Epoch: {} - Benign Time: {} - Benign Score: {} - Adversarial Time: {} - Adversarial Score: {}".format(i, ben_time, ben_score, adv_time, adv_score))
    return results, outputs

def save_results_and_outputs(results, outputs, path = "retrain/tmp") -> list:
    Path(path).mkdir(parents=True, exist_ok=True)
    for output in outputs:
        with open(f"{path}/{output}.json", "w") as f:
            json.dump(outputs[output], f)
        assert Path(f"{path}/{output}.json").exists(), f"Problem saving {path}/{output}"
    with open(f"{path}/results.json", "w") as f:
        json.dump(results, f)
    assert Path(f"{path}/results.json").exists(), f"Problem saving results to {path}/results.json"
    


attack_size = 100
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
epochs = 20
max_iter = 1000
X,y = make_classification(n_samples=12500, n_features=100, n_informative=99, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, weights=None,  class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info("Creating model")
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    model = SVC(kernel=kernel, probability=True)
    clf = ScikitlearnSVC(model=model)
    logger.info("Fitting model")
    clf.fit(X_train, y_train)
    atk = ProjectedGradientDescent(estimator=clf, eps=1, eps_step=0.1, max_iter=10, targeted=False, num_random_init=0, batch_size=attack_size)
    results, outputs = retrain_loop(clf, X_train, y_train, X_test, y_test, atk, attack_size, epochs)
    save_results_and_outputs(results, outputs, path = f"retrain/{kernel}")

    

