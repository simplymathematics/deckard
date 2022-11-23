config = """
    data:
        generated:
            n_classes: 2
            n_features: 5
            n_informative: 4
            n_redundant: 1
            n_samples: 10000
        params:
            name: classification
            random_state: 42
            shuffle: true
            stratify: true
            time_series: true
            train_noise: 0
            train_size: 8000
        files:
            data_filetype : pickle
            data_path : data
    fit:
        epochs: 1000
        learning_rate: 1.0e-08
        log_interval: 10
    model:
        params: 
            name: sklearn.ensemble.RandomForestClassifier
            n_estimators: 100
            max_depth: 5
        files:
            model_path : models
            model_filetype : pickle
    plots:
        balance: balance
        classification: classification
        confusion: confusion
        correlation: correlation
        radviz: radviz
        rank: rank
    scorers:
        accuracy:
            name: sklearn.metrics.accuracy_score
            normalize: true
        f1-macro:
            average: macro
            name: sklearn.metrics.f1_score
        f1-micro:
            average: micro
            name: sklearn.metrics.f1_score
        f1-weighted:
            average: weighted
            name: sklearn.metrics.f1_score
        precision:
            average: weighted
            name: sklearn.metrics.precision_score
        recall:
            average: weighted
            name: sklearn.metrics.recall_score
    files:
        ground_truth_file: ground_truth.json
        predictions_file: predictions.json
        time_dict_file: time_dict.json
        params_file: params.json
        score_dict_file: scores.json
        path: reports
        
    """
    yaml.add_constructor("!Experiment:", Experiment)
    experiment = yaml.load("!Experiment:\n" + str(config), Loader=yaml.Loader)
    experiment.run()