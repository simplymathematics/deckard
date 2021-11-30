from utils import load_data, checkpoint
from read_yml import parse_gridsearch_from_yml
from data import Data
from model import Model
from experiment import Experiment
from sklearn.base import BaseEstimator

if __name__ == '__main__':
    # command line arguments
    import argparse
    import os
    import logging
    import uuid
    parser = argparse.ArgumentParser(description='Run a model on a dataset')
    parser.add_argument('-m', '--model', default = 'configs/model.yml',type=str, help='Model file to use')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset file to use', default = "data.pkl")
    parser.add_argument('-o,', '--output', type=str, help='Output file to use', default = 'data')
    parser.add_argument('-f', '--folder', type=str, default = 'data', help='Folder to use', required=False)
    parser.add_argument('-v', '--verbosity', type = str, default='DEBUG', help='set python verbosity level')
    parser.add_argument('-s', '--scorer', type = str, default='f1', help='scorer for optimization. Other metrics can be set using the Experiment.set_metric method.')
    args = parser.parse_args()
    # initialize logging
    logging.basicConfig(level=args.verbosity)
    model_file = args.model
    if os.path.isdir(os.path.join(args.output, 'best_train')):
        result_file = os.path.join(args.output, 'best_train',"results.json")
    try:
         data = load_data(os.path.join('data', 'data', 'data.pkl'))
    except:
        raise ValueError("Unable to load dataset {}".format(os.path.join('data', 'data', 'data.pkl')))
    assert isinstance(data, Data)
    models = parse_gridsearch_from_yml(model_file)
    best_score = 0
    for model in models:
        assert isinstance(model, BaseEstimator)
        model_obj = Model(model)
        experiment = Experiment(data= data, model = model_obj)
        experiment.run()
        score = experiment.scores[args.scorer.upper()]
        checkpoint(filename = os.path.join("all_retrain", str(uuid.uuid4())), experiment = experiment, result_folder = args.output)
        if score > best_score:
            best_score = score
            checkpoint(filename = 'best_retrain', experiment = experiment, result_folder = args.output)
    logging.info("Best Score for trainer: {}".format(best_score))