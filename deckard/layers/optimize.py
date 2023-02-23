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
from deckard.layers.runner import load_dvc_experiment
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






@hydra.main(
    version_base=None,
    config_path=str(Path(os.getcwd(), "conf")),
    config_name="config",
)
def hydra_runner(cfg:DictConfig):
    # Stage selects a subset of the pipeline to run (i.e. number of layers to run inside a single container)
    if "stage" in cfg:
        stage = cfg.stage
        del cfg.stage
    else:
        stage = None
    if "dry_run" in cfg:
        dry_run = cfg.dry_run
        del cfg.dry_run
    else:
        dry_run = False
    if "queue" in cfg:
        queue = cfg.queue
    else:
        queue = "queue"
    if not Path(os.getcwd(), queue).exists():
        Path(os.getcwd(), queue).mkdir()
    params = OmegaConf.to_object(cfg) # This converts the hydra config to a dictionary
    params = parse(params) # This is a hack to add file names based on the hash of the parameterization
    logger.info("Params:\n"+json.dumps(params, indent=4)) # For debugging
    filename = Path(os.getcwd(), queue, my_hash(params)+".yaml") # This is the file that will be used to run the experiment
    with open(filename, 'w') as f:
        yaml.dump(params, f)
    if not dry_run: #If dry_run is true, this will just write the parameters to a file and not run the experiment
        exp =  load_dvc_experiment(stage=stage, params=params)
        ########################################
        # For # This will run the experiment and return a dictionary of results. 
        # This uses the normal fit/predict/eval loop and returns the scores 
        # on the test set as specified in the config file. 
        # So, essentially, we would a function that takes in the parameters and returns the score here.
        results = exp.run() 
        ########################################
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
        logger.info("Time:\n"+json.dumps(time, indent=4))
    else:
        score = 0
    return score



if '__main__' == __name__:
    hydra_runner()