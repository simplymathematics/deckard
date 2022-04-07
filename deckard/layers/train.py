

if __name__ == '__main__':
    import logging
    from deckard.base import Data
    from deckard.base.utils import load_data, save_all, save_best_only
    from deckard.base import Data
    from deckard.base.parse import parse_list_from_yml, generate_object_list, transform_params_for_pipeline, generate_experiment_list
    from os import path, mkdir
    import argparse
    parser = argparse.ArgumentParser(description='Run a model on a dataset')
    parser.add_argument('-c', '--config', default = 'configs/model.yml',type=str, help='config file to use')
    parser.add_argument('-f', '--folder', type=str, help='Experiment folder to use', default = './')
    parser.add_argument('-d', '--dataset', type=str, help='Data file to use', default = "data.pkl")
    parser.add_argument('-b' ,'--bigger_is_better', type = bool, default = True, help='whether the scorer is bigger is better')
    parser.add_argument('-v', '--verbosity', type = str, default='DEBUG', help='set python verbosity level')
    parser.add_argument('-s', '--scorer', default = 'f1', type = str, help='scorer for optimization. Other metrics can be set using the Experiment.set_metric method.')
    parser.add_argument('--best', type=bool, default = False, help='only store the best model')
    parser.add_argument('--time_series', type = bool, default = False, help = "Whether to use time series")
    parser.add_argument('--name', type=str, default = "train", help='name of the experiment')
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    filename = args.config
    try:
         data = load_data(path.join(args.folder, args.dataset))
         if args.time_series == True:
             data.time_series = True
    except:
        raise ValueError("Unable to load dataset {}".format(path.join(args.folder, args.dataset)))
    assert isinstance(data, Data)
    if not path.exists(args.folder):
        mkdir(args.folder)
    # reads the config file
    model_list = parse_list_from_yml(args.config)
    # instantiates those objects
    model_list = generate_object_list(model_list)
    # turns lists of params into a set of permutations where len(permutations) = len(list1) * len(list2) ... len(listn)
    model_list = transform_params_for_pipeline(model_list, 'model')
    # initalizes the experiment objects using the above data and models
    # Change the default scorer with Experiment.set_metric_scorer by passing scorer= during instantiation, or specifying it as a model parameter in the config file
    # Eventually, scoring will rely on the same yaml configs for the sake of consistency
    # For now, the defaults detect whether the estimator is a classifier or a regressor, using f1 and R2 as the objective measures respectively, while reporting several other common metrics
    # Fit and predict time are always reported (if available)
    exp_list = generate_experiment_list(model_list, data, cv = 10)
    scorer = args.scorer.upper()
    if args.best:
        save_best_only(folder=args.folder, exp_list=exp_list, scorer=scorer, bigger_is_better=args.bigger_is_better, name=args.name)
    else:
        save_all(folder=args.folder, exp_list=exp_list, scorer=scorer, bigger_is_better=args.bigger_is_better, name=args.name)
   
