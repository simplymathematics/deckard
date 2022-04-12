import logging
import pickle
import os
from deckard.base.model import Model
from deckard.base.data import Data
from art.estimators.classification import PyTorchClassifier, KerasClassifier, TensorFlowClassifier
from art.estimators.classification.scikitlearn import SklearnClassifier
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

def load_model(filename:str = None, path:str = ".", mtype:str =None) -> Model:
    """
    Load a model from a pickle file.
    filename: the pickle file to load the model from
    mtype: the type of model to load
    """
    from deckard.base.model import Model
    logger.debug("Loading model")
    # load the model
    filename = os.path.join(path, filename)
    if filename.endswith('.pkl') or mtype == 'sklearn' or 'pipeline' or 'gridsearch' or 'pickel':
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        model = Model(model)
    elif mtype == 'keras' or filename.endswith('.h5'):
        from tensorflow.keras.models import load_model as keras_load_model
        # TODO add support for tf, pytorch
        cloned_model = keras_load_model(filename)
        model = Model(cloned_model)
    elif 'mtype' =='torch' or 'pytorch':
        from torch import load as torch_load
        model = torch_load(filename)
        model = Model(model)
    elif 'mtype' == 'tf' or 'tensorflow':
        from tensorflow.keras.models import load_model as tf_load_model
        model = tf_load_model(filename)
        model = Model(model)
    else:
        raise TypeError("Model type {} is not supported".format(mtype))
    logger.info("Loaded model")
    return model

def load_data(filename:str = None, path:str=".") -> Data:
    """
    Load a data file.
    data_file: the data file to load
    """
    data_file = os.path.join(path, filename)
    from deckard.base.data import Data
    logger.debug("Loading data")
    # load the data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    assert isinstance(data, Data), "Data is not an instance of Data. It is type: {}".format(type(data))
    logger.info("Loaded model")
    return data

SUPPORTED_DEFENSES = [Postprocessor, Preprocessor, Transformer, Trainer]
SUPPORTED_MODELS = [PyTorchClassifier, SklearnClassifier, KerasClassifier, TensorFlowClassifier]

def initialize_art_classifier(filename:str, path:str = None, model_type:str=None, url:str = None, output_dir:str = None,):
    """
    Load an ART model.
    :param model_name: the name of the model
    :param model_path: the path to the model
    :param model_type: the type of the model
    :param output_dir: the output folder to save the model to
    :return: the loaded art model
    
    """
    # Download/load model
    assert model_type is not None, "model_type must be specified"
    assert filename is not None, "filename must be specified"
    assert path is not None or url is not None, "path or url must be specified"
    if output_dir == None:
        output_dir = os.path.dirname(path)
    if url is not None:
        # download model
        model_path = get_file(filename = filename, extract=False, path=path, url=url, verbose = True)
    else:
        model_path = os.path.join(path, filename)
    # Define type for ART
    if model_type == 'tfv1' or 'tensorflowv1' or 'tf1':
        import tensorflow.compat.v1 as tfv1
        tfv1.disable_eager_execution()
        from tensorflow.keras.models import load_model as keras_load_model
        classifier_model = keras_load_model(model_path)
        art_model = TensorFlowClassifier( model=classifier_model)
    elif model_type == 'keras' or 'k':
        from tensorflow.keras.models import load_model as keras_load_model
        classifier_model = keras_load_model(model_path)
        art_model = KerasClassifier( model=classifier_model)
    elif model_type == 'tf' or 'tensorflow':
        from tensorflow.keras.models import load_model as keras_load_model
        classifier_model = keras_load_model(model_path)
        art_model = TensorFlowClassifier( model=classifier_model)
    elif model_type == 'pytorch' or 'torch':
        raise NotImplementedError("Pytorch not implemented yet")
    elif model_type == 'sklearn' or 'scikit-learn':
        from pickle import load
        with open(model_path, 'rb') as f:
            classifier_model = load(f)
        art_model = SklearnClassifier(model=classifier_model)
    else:
        raise ValueError("Unknown model type {}".format(model_type))
    return art_model


def loggerCall():
    logger = logging.getLogger(__name__)
    logger.debug('SUBMODULE: DEBUG LOGGING MODE : ')
    logger.info('Submodule: INFO LOG')
    return logger

def save_best_only(path:str, exp_list:list, scorer:str, bigger_is_better:bool,  best_score:float = None):
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
            exp.run()
            if exp.scores[scorer] >= best_score and bigger_is_better:
                best = exp
            else:
                pass

        if not os.path.isdir(path):
            os.mkdir(path)
        best.save_model(path = path)
        best.save_results(path = path)
        logger.info("Saved best experiment to {}".format(path))
        logger.info("Best score: {}".format(best.scores[scorer]))

def save_all(path:str, exp_list:list, scorer:str, bigger_is_better:bool, best_score:float=None):
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
        if best_score == None and bigger_is_better:
            best_score = -1e10
        elif best_score == None and not bigger_is_better:
            best_score = 1e10
        for exp in exp_list:
            exp.run()
            if not os.path.isdir(os.path.join(path, exp.filename)):
                os.mkdir(os.path.join(path, exp.filename))
                logger.info("Created path: " + os.path.join(path, exp.filename))
            exp.save_results(path = os.path.join(path, exp.filename))
            exp.save_model(path = os.path.join(path, exp.filename))
            if exp.scores[scorer] >= best_score and bigger_is_better:
                best_score = exp.scores[scorer]
                best = exp
        best.save_model(path = path)
        best.save_results(path = path)
        logger.info("Saved best experiment to {}".format(path))
        logger.info("Best score: {}".format(best.scores[scorer]))