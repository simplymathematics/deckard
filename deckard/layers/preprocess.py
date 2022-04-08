if __name__ == '__main__':
    import logging
    from deckard.base import Experiment
    from deckard.base.utils import save_all, save_best_only, load_data, load_model
    from deckard.base.parse import parse_list_from_yml, generate_object_list, transform_params_for_pipeline, insert_layer_into_pipeline, generate_sklearn_experiment_list
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
    parser.add_argument('--name', type=str, help='name of the experiment', required=True)
    parser.add_argument('--experiment', type=str,  help='name of the experiment', required=True)
    parser.add_argument('-p', '--position', type = int, default = -1, help='position of the preprocessor in the pipeline. Useful for using more than one pre-processor.')
    parser.add_argument('--best', type=bool, default=False, help='only store the best preprocessor')
    parser.add_argument('--model_name', type=str, default = "model.pkl", help='name of the experiment')
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    preprocessor_file = args.config
    from deckard.base.utils import load_model
    best = load_model(path.join(args.folder, args.experiment, args.model_name))
    data = load_data(path.join(args.folder, args.dataset))
    preprocessor_list = parse_list_from_yml(args.config)
    object_list = generate_object_list(preprocessor_list)
    transformed_list = transform_params_for_pipeline(object_list, args.name)
    model_list = insert_layer_into_pipeline(transformed_list, best.model.estimator, name = args.name, position = args.position)
    exp_list = generate_sklearn_experiment_list(model_list, data, cv = 10)
    scorer = args.scorer.upper()
    folder = path.join(args.folder, args.name)
    if args.best:
        save_best_only(folder = args.folder, exp_list = exp_list, scorer=scorer, bigger_is_better=args.bigger_is_better, name = args.name)
    else:
        save_all(folder = args.folder, exp_list = exp_list, scorer=scorer, bigger_is_better=args.bigger_is_better, name = args.name)