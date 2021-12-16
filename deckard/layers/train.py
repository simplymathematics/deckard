

if __name__ == '__main__':
    import logging
    from deckard.base import Data
    from deckard.base.utils import load_data, return_result
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
    parser.add_argument('--time_series', type = bool, default = False, help = "Whether to use time series")
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    model_file = args.config
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
    # turns lists of params into a set of permutations
    model_list = transform_params(model_list, 'model')
    # initalizes the experiment objects using the above data and models
    exp_list = generate_experiment_list(model_list, data)
    scorer = args.scorer.upper()
    folder = path.join(args.folder, 'best_train')
    flag = False # does a best model exist yet?
    for exp in exp_list:
        exp.run()
        exp.save_results(folder) #cache all reults
        if flag == False:
            best = exp
            flag = True
        elif exp.scores[scorer] >= best.scores[scorer] and args.bigger_is_better:
            best = exp # only save binaries if they outperform previous models, overwriting said model
    best.save_experiment(folder)
   
