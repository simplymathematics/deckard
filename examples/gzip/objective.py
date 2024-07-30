import optuna
from gzip_classifier import all_metrics


def objective(trial: optuna.Trial):
    model_type = trial.suggest_categorical("model_type", ["knn", "logistic", "svc"])
    metric = trial.suggest_categorical("model.init.metric", all_metrics.keys())
    if model_type == "knn":
        k = trial.suggest_categorical("k", [3, 5, 7, 9, 11])
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        algorithm = trial.suggest_categorical("algorithm", ["brute"])
        params = {"k": k, "weights": weights, "algorithm": algorithm}
    elif model_type == "logistic":
        C = trial.suggest_loguniform("C", 1e-10, 1e10)
        solver = trial.suggest_categorical("solver", ["saga"])
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", None])
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
        params = {
            "C": C,
            "solver": solver,
            "penalty": penalty,
            "fit_intercept": fit_intercept,
            "class_weight": class_weight,
        }
    elif model_type == "svc":
        C = trial.suggest_loguniform("C", 1e-10, 1e10)
        kernel = trial.suggest_categorical(
            "kernel", ["linear", "rbf", "poly", "sigmoid"]
        )
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
        if kernel == "poly":
            degree = trial.suggest_int("degree", 2, 5)
            params = {
                "C": C,
                "kernel": kernel,
                "degree": degree,
                "class_weight": class_weight,
            }
        elif kernel == "rbf":
            gamma = trial.suggest_categorical("gamma", ["auto", "scale"])
            params = {
                "C": C,
                "kernel": kernel,
                "gamma": gamma,
                "class_weight": class_weight,
            }
        else:
            params = {"C": C, "kernel": kernel, "class_weight": class_weight}
    else:
        raise NotImplementedError(f"Model type {model_type} not supported.")
    params["metric"] = metric
    params["model_name"] = f"{metric}_{model_type}"
