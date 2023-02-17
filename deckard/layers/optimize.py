import argparse
import logging
import os
import subprocess
import yaml
import json
from pathlib import Path
import optuna
import hydra

from omegaconf import DictConfig, OmegaConf
from deckard.base import Experiment
from deckard.layers.runner import run_dvc_experiment
from deckard.layers.parse import parse
from deckard.base.hashable import my_hash


logger = logging.getLogger(__name__)

# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.utils import gen_batches

# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# classes = np.unique(y)
# n_train_iter = 10000
# length = len(X_train)


# def objective(trial):
#     alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
#     loss = trial.suggest_categorical('loss', ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'])
#     penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
#     clf = SGDClassifier(alpha=alpha, loss=loss, penalty=penalty)

#     for step in range(n_train_iter):
#         clf.partial_fit(X_train, y_train, classes=classes)
#         intermediate_value = clf.score(X_test, y_test)
#         trial.report(intermediate_value, step)
#         if trial.should_prune():
#             raise optuna.TrialPruned()
#     return clf.score(X_test, y_test)




# if "__main__" == __name__:
#     study = optuna.create_study(
#     direction='maximize',
#     pruner=optuna.pruners.HyperbandPruner(
#         min_resource=1,
#         max_resource='auto',
#         reduction_factor=3,
#     )
# )
#     study.optimize(objective, n_trials=1000, njobs = -1)
#     optuna.visualization.plot_contour(study, params=['alpha', 'loss'])

@hydra.main(
    version_base=None,
    config_path=Path(os.getcwd(), "conf"),
    config_name="config",
)
def hydra_runner(cfg:DictConfig):
    if "stage" in cfg:
        stage = cfg.stage
        del cfg.stage
    else:
        stage = None
    params = OmegaConf.to_object(cfg)
    params = parse(params)
    logger.info("Params:\n"+json.dumps(params, indent=4))
    filename = Path(os.getcwd(), "queue", my_hash(params)+".yaml")
    with open(filename, 'w') as f:
        yaml.dump(params, f)
    bashCommand = "dvc pull"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    logger.info(output)
    logger.error(error)
    results =  run_dvc_experiment(params = params, stage=stage)
    bashCommand = "dvc push"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    results = results[stage] if stage is not None else list(results.values())[-1]
    logger.info("Results:\n"+json.dumps(results, indent=4))
    with open(results['scores'], 'r') as f:
        score = yaml.load(f, Loader=yaml.FullLoader)
    logger.info("Score:\n"+json.dumps(score, indent=4))
    score = list(score.values())[0]
    logger.info("Score:\n"+json.dumps(score, indent=4))
    with open(results['time'], 'r') as f:
        time = yaml.load(f, Loader=yaml.FullLoader)
    logger.info("Time:\n"+json.dumps(time, indent=4))
    time = list(time.values())[0]
    return score, time

if '__main__' == __name__:
    hydra_runner()