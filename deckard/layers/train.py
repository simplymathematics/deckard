

if __name__ == '__main__':
    import logging
    from deckard.base import Data
    from deckard.base.utils import load_data, save_all, save_best_only
    from deckard.base import Data
    from deckard.base.parse import parse_list_from_yml, generate_object_list, transform_params, generate_experiment_list
    from os import path, mkdir
    import argparse
    parser = argparse.ArgumentParser(description='Run a model on a dataset')
    parser.add_argument('-c', '--config', default = 'configs/model.yml',type=str, help='config file to use')
    parser.add_argument('-f', '--folder', type=str, help='Experiment folder to use', default = './')
    parser.add_argument('-d', '--dataset', type=str, help='Data file to use', default = "data.pkl")
    parser.add_argument('-b' ,'--bigger_is_better', type = bool, default = True, help='whether the scorer is bigger is better')
    parser.add_argument('-v', '--verbosity', type = str, default='DEBUG', help='set python verbosity level')
    parser.add_argument('-s', '--scorer', default = 'f1', type = str, help='scorer for optimization. Other metrics can be set using the Experiment.set_metric method.')
    parser.add_argument('--best', type=bool, default = "True", help='only store the best model')
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    model_file = args.config
    try:
         data = load_data(path.join(args.folder, args.dataset))
    except:
        raise ValueError("Unable to load dataset {}".format(path.join(args.folder, args.dataset)))
    assert isinstance(data, Data)
    if not path.exists(args.folder):
        mkdir(args.folder)
    model_list = parse_list_from_yml(args.config)
    model_list = generate_object_list(model_list)
    model_list = transform_params(model_list, 'model')
    exp_list = generate_experiment_list(model_list, data)
    scorer = args.scorer.upper()
    folder = path.join(args.folder, 'best_train')
    
    if args.best:
        save_best_only(folder=args.folder, exp_list=exp_list, scorer=scorer, bigger_is_better=args.bigger_is_better)
    else:
        save_all(folder=args.folder, exp_list=exp_list, scorer=scorer, bigger_is_better=args.bigger_is_better)
   