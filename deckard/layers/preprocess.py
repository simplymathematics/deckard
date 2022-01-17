if __name__ == '__main__':
    import logging
    from deckard.base import Experiment
    from deckard.base.utils import  load_experiment, save_all, save_best_only
    from deckard.base.parse import parse_list_from_yml, generate_object_list, transform_params, insert_layer_into_list, generate_experiment_list
    import argparse
    from os import path
    import logging
    parser = argparse.ArgumentParser(description='Run a preprocessor on a dataset')
    parser.add_argument('-c', '--config', default = 'configs/preprocess.yml',type=str, help='preprocessor file to use')
    parser.add_argument('-f', '--folder', type=str, help='Experiment folder to use', default = './')
    parser.add_argument('-d', '--dataset', type=str, help='Data file to use', default = "data.pkl")
    parser.add_argument('-b' ,'--bigger_is_better', required=True, default = True, help='whether the scorer is bigger is better')
    parser.add_argument('-v', '--verbosity', type = str, default='DEBUG', help='set python verbosity level')
    parser.add_argument('-s', '--scorer', required=True, type = str, help='scorer for optimization. Other metrics can be set using the Experiment.set_metric method.')
    parser.add_argument('-p', '--position', type = int, default = -1, help='position of the preprocessor in the list')
    parser.add_argument('--best', action='store_true', help='only store the best preprocessor')
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    preprocessor_file = args.config
    best = load_experiment(path.join(args.folder, 'best_preprocess', 'experiment.pkl'))
    assert isinstance(best, Experiment)
    preprocessor_list = parse_list_from_yml(args.config)
    preprocessor_list = generate_object_list(preprocessor_list)
    preprocessor_list = transform_params(preprocessor_list, 'preprocess')
    model_list = insert_layer_into_list(preprocessor_list, best.model.model.estimator, name = 'preprocess', position = args.position)
    exp_list = generate_experiment_list(model_list, best.data)
    scorer = args.scorer.upper()
    folder = path.join(args.folder, 'best_preprocess')
    if args.best:
        save_best_only(args.folder, exp_list, scorer=scorer, bigger_is_better=args.bigger_is_better)
    else:
        save_all(args.folder, exp_list)