

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
    # turns lists of params into a set of permutations where len(permutations) = len(list1) * len(list2) ... len(listn)
    model_list = transform_params(model_list, 'model')
    # initalizes the experiment objects using the above data and models
    # Change the default scorer with Experiment.set_metric_scorer by passing scorer= during instantiation, or specifying it as a model parameter in the config file
    # Eventually, scoring will rely on the same yaml configs for the sake of consistency
    # For now, the defaults detect whether the estimator is a classifier or a regressor, using f1 and R2 as the objective measures respectively, while reporting several other common metrics
    # Fit and predict time are always reported (if available)s
    exp_list = generate_experiment_list(model_list, data)
    scorer = args.scorer.upper()
    folder = path.join(args.folder, 'best_train')
    flag = False # does a best model exist yet?
    for exp in exp_list: #iterates through each object specified in the config file, passes hyper-parameter optimization to GridSearchCV and optimizes for the default metrics. 
        exp.run()
        exp.save_results(folder) #cache all reults
        if flag == False:
            best = exp
            flag = True
            
        elif exp.scores[scorer] >= best.scores[scorer] and args.bigger_is_better: # user specified scoring function from results. 
#             Divorcing this from the grid search optimization allows for multi-objective optimization or distributed training. For example, we can select
#             accuracy as our optimization criteria during training, but choose a model based on, for example bandwidth or processor constraints later.
#             In this way, we can have a primary and secondary objective, which is particularly useful in the context of overdetermined systmes in which
#             multiple configurations can lead to the same failure rate but have other run-time or robustness characteristics that make one option preferable.
            best = exp # only save binaries if they outperform previous models, overwriting said model
    best.save_experiment(folder)
   
