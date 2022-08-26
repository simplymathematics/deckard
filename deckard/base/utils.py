import os, json, logging, pickle
from deckard.base.model import Model
from deckard.base.data import Data
from art.estimators.classification import PyTorchClassifier, KerasClassifier, TensorFlowClassifier, SklearnClassifier
from art.estimators import ScikitlearnEstimator
from art.defences.preprocessor import Preprocessor
from art.defences.postprocessor import Postprocessor
from art.defences.trainer import Trainer
from art.defences.transformer import Transformer
from art.utils import get_file
logger = logging.getLogger(__name__)

def return_score(scorer:str, filename = 'results.json', path:str=".")-> float:
    """
    Return the result of the experiment.
    scorer: the scorer to use
    filename: the filename to load the results from
    path: the path to the results
    """
    import json
    filename = os.path.join(path, filename)
    assert os.path.isfile(filename), "{} does not exist".format(filename)
    # load json
    with open(filename) as f:
        results = json.load(f)
    # return the result
    return results[scorer.upper()]

def find_successes(input_folder, filename:str, dict_name:str = None):
        failures = []
        successes = []
        for folder in os.listdir(input_folder):
            if os.path.isdir(os.path.join(input_folder, folder)):
                files = os.listdir(os.path.join(input_folder, folder))
                if 'scores.json' or 'adversarial_scores.json' in files:
                    if dict_name is not None:
                        with open(os.path.join(input_folder, folder, filename)) as f:
                            model_params = json.load(f)[dict_name]
                    else:
                        with open(os.path.join(input_folder, folder, filename)) as f:
                            model_params = json.load(f)
                    model_name = model_params['name']
                    model_params = model_params['params']
                    successes.append((model_name, model_params))
                else:
                    failures.append(folder)
            else:
                files = os.listdir(input_folder)
                if 'scores.json' or 'adversarial_scores.json' in files:
                    if dict_name is not None:
                        with open(os.path.join(input_folder, filename)) as f:
                            model_params = json.load(f)[dict_name]                       
                    else:
                        with open(os.path.join(input_folder, filename)) as f:
                            model_params = json.load(f)
                    if 'Name' in model_params.keys():
                        model_name = model_params['Name']
                    else:
                        model_name = str(type(model_params['Model'])).split("'")[1]
                    if 'params' in model_params.items():
                        model_params = model_params['params']
                    else:
                        model_params = model_params
                    successes.append((model_name, model_params))
                else:
                    failures.append(folder)           
        return successes, failures

def remove_successes_from_queue(successes, todos):
    completed_tasks = []
    for model_name, model_params in successes:
        i = 0
        for queue_name, queue_params in todos:
            queue_name = queue_name.split(".")[-1]
            i += 1
            if model_name.split("'")[1].split(".")[-1] == queue_name and queue_params.items() <= model_params.items():
                completed_tasks.append(i)    
    todos = [i for j, i in enumerate(todos) if j not in completed_tasks]
    return todos
    


SUPPORTED_DEFENSES = (Postprocessor, Preprocessor, Transformer, Trainer)
SUPPORTED_MODELS = (PyTorchClassifier, ScikitlearnEstimator, KerasClassifier, TensorFlowClassifier)




def loggerCall():
    logger = logging.getLogger(__name__)
    logger.debug('SUBMODULE: DEBUG LOGGING MODE : ')
    logger.info('Submodule: INFO LOG')
    return logger

def save_best_only(path:str, exp_list:list, scorer:str, bigger_is_better:bool,  best_score:float = None, name:str = None):
        """
        Save the best experiment only.
        path: the path to save the experiment
        exp_list: the list of experiments
        scorer: the scorer to use
        bigger_is_better: if True, the best experiment is the one with the highest score, otherwise the best is the one with the lowest score
        name: the name of the experiment
        """
        if best_score == None and bigger_is_better:
            best_score = -1e10
        elif best_score == None and not bigger_is_better:
            best_score = 1e10
        for exp in exp_list:
            exp.run(path)
            if exp.scores[scorer] >= best_score and bigger_is_better:
                best = exp
            else:
                pass
        if not os.path.isdir(path):
            os.mkdir(path)
        best.save_model(filename = 'model', path = path)
        best.save_results(path = path)
        best.save_params(path = path)
        logger.info("Saved best experiment to {}".format(path))
        logger.info("Best score: {}".format(best.scores[scorer]))

def save_all(path:str, exp_list:list, scorer:str, bigger_is_better:bool, best_score:float=None, name:str = None):
        """
        Runs and saves all experiments.
        path: the path to save the experiments
        exp_list: the list of experiments
        scorer: the scorer to use
        bigger_is_better: if True, the best experiment is the one with the highest score, otherwise the best is the one with the lowest score
        """
        if not os.path.isdir(path):
            os.mkdir(path)
            logger.info("Created path: " + path)
        if not os.path.isdir(os.path.join(path, name)):
            os.mkdir(os.path.join(path, name))
            logger.info("Created path: " + os.path.join(path, name))
        path = os.path.join(path,name)
        if best_score == None and bigger_is_better:
            best_score = -1e10
        elif best_score == None and not bigger_is_better:
            best_score = 1e10
        for exp in exp_list:
            exp.filename = str(exp.filename)
            if path is not None and not os.path.isdir(os.path.join(path, exp.filename)):
                os.mkdir(os.path.join(path, exp.filename))
                logger.info("Created path: " + os.path.join(path, exp.filename))
            exp.run(os.path.join(path, exp.filename))
            exp.save_results(path = os.path.join(path, exp.filename))
            exp.save_model(filename = 'model', path = os.path.join(path, exp.filename))
            if exp.scores[scorer] >= best_score and bigger_is_better:
                best_score = exp.scores[scorer]
                best = exp
        best.save_model(filename = 'model', path = path)
        best.save_results(path = path)
        best.save_params(path = path)
        logger.info("Best score: {}".format(best.scores[scorer]))