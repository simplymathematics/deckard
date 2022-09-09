if __name__ == '__main__':
    import logging
    from deckard.base import Data
    from .parse import generate_object_list_from_tuple, generate_tuple_list_from_yml, generate_experiment_list
    from .utils import  save_all, save_best_only
    from os import path, mkdir
    import argparse
    parser = argparse.ArgumentParser(description='Run a model on a dataset')
    parser.add_argument('-c', '--config', default = 'configs/model.yml',type=str, help='config file to use')
    parser.add_argument('-f', '--folder', type=str, help='Experiment folder to use', default = './')
    parser.add_argument('-d', '--dataset', type=str, help='Data file to use', default = "data.pkl")
    parser.add_argument('-b' ,'--bigger_is_better', type = bool, default = True, help='whether the scorer is bigger is better')
    parser.add_argument('-s', '--scorer', default = 'f1', type = str, help='scorer for optimization. Other metrics can be set using the Experiment.set_metric method.')
    parser.add_argument('--best', type=bool, default = False, help='only store the best model')
    parser.add_argument('--time_series', type = bool, default = False, help = "Whether to use time series")
    parser.add_argument('--name', type=str, default = "train", help='name of the experiment')
    args = parser.parse_args()
    filename = args.config
    try:
         data = Data(path.join(args.folder, args.dataset))
         if args.time_series == True:
             data.time_series = True
    except:
        raise ValueError("Unable to load dataset {}".format(path.join(args.folder, args.dataset)))
    assert isinstance(data, Data)
    if not path.exists(args.folder):
        mkdir(args.folder)
    # reads the configs, generating a set of tuples such that
    # each tuple is a combination of parameters a given estimator
    # the length of the list is = len(list of estimators)*len(param_1)*len(param_2)*...len(param_n)
    tuple_list = generate_tuple_list_from_yml(args.config)
    model_list = generate_object_list_from_tuple(tuple_list)
    exp_list = generate_experiment_list(model_list, data)
    scorer = args.scorer.upper()
    if args.best:
        save_best_only(path=args.folder, exp_list=exp_list, scorer=scorer, bigger_is_better=args.bigger_is_better, name=args.name)
    else:
        save_all(path=args.folder, exp_list=exp_list, scorer=scorer, bigger_is_better=args.bigger_is_better, name=args.name)
   
