
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from deckard.base import Model, Data
from copy import deepcopy
logger = logging.getLogger(__name__)
if __name__ == '__main__':
    import logging
    from deckard.base.utils import save_all, save_best_only
    from deckard.base.parse import generate_experiment_list, generate_object_list_from_tuple, generate_tuple_list_from_yml
    import argparse
    from os import path
    parser = argparse.ArgumentParser(description='Run a preprocessor on a dataset')
    parser.add_argument('-c', '--config', default = 'configs/preprocess.yml',type=str, help='preprocessor file to use')
    parser.add_argument('-f', '--folder', type=str, help='Experiment folder to use', default = './')
    parser.add_argument('-d', '--dataset', type=str, help='Data file to use', default = "data.pkl")
    parser.add_argument('-b' ,'--bigger_is_better', required=True, default = True, help='whether the scorer is bigger is better')
    parser.add_argument('-s', '--scorer', required=True, type = str, help='scorer for optimization. Other metrics can be set using the Experiment.set_metric method.')
    parser.add_argument('--name', type=str, help='name of the experiment', required=True)
    parser.add_argument('--input', type=str,  help='name of the experiment', required=True)
    parser.add_argument('-p', '--position', type = int, default = 0, help='position of the preprocessor in the pipeline. Useful for using more than one pre-processor.')
    parser.add_argument('--best', type=bool, default=False, help='only store the best preprocessor')
    parser.add_argument('--model_name', type=str, default = "model", help='name of the experiment')
    args = parser.parse_args()
    preprocessor_file = args.config
    best = Model(path.join(args.folder, args.input, args.model_name))
    data = Data(path.join(args.folder, args.dataset))
    tuple_list = generate_tuple_list_from_yml(args.config)
    preprocessor_list = generate_object_list_from_tuple(tuple_list)
    model_list = []
    model = Model(path.join(args.folder, args.input, args.model_name))
    if isinstance(model.model, (BaseEstimator, TransformerMixin)) and not isinstance(model.model, Pipeline):
        model.model = Pipeline([('model', model.model)])
    for preprocessor in preprocessor_list:
        new_model = deepcopy(model)
        try:
            new_model.model.steps.insert(args.position, (args.name, preprocessor))
        except AttributeError:
            logging.warning("Cannot add preprocessor to initialized ART estimator. Attempting to add to pipeline. Defenses should be applied after this step as they have been removed.")
            new_model = Pipeline([('model', new_model.model.model)])
            new_model.steps.insert(args.position, (args.name, preprocessor))
            new_model = Model(new_model)
        model_list.append(new_model)

    
    exp_list = generate_experiment_list(model_list, data)
    scorer = args.scorer.upper()
    if args.best:
        save_best_only(path = args.folder, exp_list = exp_list, scorer=scorer, bigger_is_better=args.bigger_is_better, name = args.name)
    else:
        save_all(path = args.folder, exp_list = exp_list, scorer=scorer, bigger_is_better=args.bigger_is_better, name = args.name)