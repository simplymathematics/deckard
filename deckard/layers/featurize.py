if __name__ == '__main__':
    import logging
    from deckard.base import Experiment
    from deckard.base.utils import load_data, load_experiment, load_model, return_result
    from deckard.base.parse import parse_list_from_yml, generate_object_list, transform_params, insert_layer_into_list, generate_experiment_list
    from os import path, mkdir
    import argparse
    import os
    import logging
    parser = argparse.ArgumentParser(description='Run a featurizer on a dataset')
    parser.add_argument('-c', '--config', default = 'configs/featurize.yml',type=str, help='featurizer file to use')
    parser.add_argument('-f', '--folder', type=str, help='Experiment folder to use', default = './')
    parser.add_argument('-d', '--dataset', type=str, help='Data file to use', default = "data.pkl")
    parser.add_argument('-b' ,'--bigger_is_better', required=True, default = True, help='whether the scorer is bigger is better')
    parser.add_argument('-v', '--verbosity', type = str, default='DEBUG', help='set python verbosity level')
    parser.add_argument('-s', '--scorer', required=True, type = str, help='scorer for optimization. Other metrics can be set using the Experiment.set_metric method.')
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    featurizer_file = args.config
    best = load_experiment(os.path.join(args.folder, 'best_preprocess', 'experiment.pkl'))
    assert isinstance(best, Experiment)
    featurizer_list = parse_list_from_yml(args.config)
    featurizer_list = generate_object_list(featurizer_list)
    featurizer_list = transform_params(featurizer_list, 'featurize')
    model_list = insert_layer_into_list(featurizer_list, best.model.model.estimator, name = 'featurize', position = 0)
    exp_list = generate_experiment_list(model_list, best.data)
    scorer = args.scorer.upper()
    folder = path.join(args.folder, 'best_features')
    flag = False
    for exp in exp_list:
        exp.run()
        exp.save_results(folder)
        if flag == False:
            best = exp
            flag = True
        elif exp.scores[scorer] >= best.scores[scorer] and args.bigger_is_better:
            best = exp
    best.save_experiment(folder)