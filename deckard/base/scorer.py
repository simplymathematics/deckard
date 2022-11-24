import json
import logging
import os
from pathlib import Path
from hashable import BaseHashable, my_hash
import pandas as pd
import yaml
import collections
from utils import factory

logger = logging.getLogger(__name__)


class Scorer(
    collections.namedtuple(
        typename="Scorer",
        field_names="data, model, scorers, plots, files",
        defaults=({}, {}, {}, {}, {}),
        rename=True,
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    def read_data_from_json(self, json_file: str):
        """Read data from json file."""
        assert os.path.isfile(json_file), "File {} does not exist.".format(json_file)
        data = pd.read_json(json_file, typ="series")
        return data

    def read_score_from_json(self, name: str, score_file: str):
        """Read score from score file."""
        assert hasattr(self, "names"), "Scorer must be initialized with a name."
        with open(score_file, "r") as f:
            score_dict = json.load(f)
        logger.info("Score read from score file {}.".format(score_file))
        return score_dict[name]

    def score(self, ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> None:
        """
        Sets scorers for evalauation if specified, returns a dict of scores in general.
        """
        scores = {}
        # if predictions.shape != ground_truth.shape:
        #     raise ValueError("Predictions and ground truth must have the same shape.")
        names = self.scorers.keys()
        scorers = [self.scorers[name] for name in names]
        for name, scorer in zip(names, scorers):
            score = factory(
                scorer.pop("name"), **scorer, y_pred=predictions, y_true=ground_truth
            )
            scores[name] = score
        scores = pd.Series(scores).T
        return scores

    def get_name(self):
        """Return the names of the scorer."""
        names = self.scorers.keys()
        logger.info("Returning names {}.".format(names))
        return names

    def get_scorers(self):
        """
        Sets the scorer for an experiment
        :param experiment: experiment to set scorer for
        :param scorer: scorer to set
        """
        return str(self)

    def save_score(
        self,
        results,
        filename: str = "scores.json",
    ) -> None:
        """
        Saves scores to specified file.
        :param filename: str, names of file to save scores to.
        :param path: str, path to folder to save scores. If none specified, scores are saved in current working directory. Must exist.
        """
        score_file = filename
        if not isinstance(results, pd.Series):
            results = pd.Series(results.values(), name=filename, index=results.keys())
        results.to_json(score_file)
        assert os.path.exists(score_file), "Score file not saved"
        return results

    def save_list_score(
        results: dict,
        filename: str = "scores.json",
    ) -> None:
        """
        Saves scores to specified file.
        :param filename: str, names of file to save scores to.
        :param path: str, path to folder to save scores. If none specified, scores are saved in current working directory. Must exist.
        """
        score_file = filename
        filetype = filename.split(".")[-1]
        try:
            results = pd.DataFrame(
                results.values(),
                names=score_file,
                index=results.keys(),
            )
        except TypeError as e:
            if "unexpected keyword argument 'name'" in str(e):
                results = pd.DataFrame(results.values(), index=results.keys())
            else:
                raise e
        if filetype == "json":
            results.to_json(score_file)
        elif filetype == "csv":
            results.to_csv(score_file)
        else:
            raise NotImplementedError("Filetype {} not implemented.".format(filetype))
        assert os.path.exists(score_file), "Score file not saved"
        return results

    def save_results(self, scores: dict, filename: str, path: str = ".") -> None:
        """
        Saves all data to specified folder, using default filenames.
        """
        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except FileExistsError:
                logger.warning("Path {} already exists. Overwriting".format(path))
        save_names = []
        save_scores = []
        results = {}
        names = self.scorers.keys()
        for name, score in zip(names, scores):
            results[name] = score
            if isinstance(score, (list, tuple)):
                filename = name
                result = self.save_list_score(
                    {name: score},
                    filename=filename,
                )
                results[name] = result
            else:
                save_names.append(name)
                save_scores.append(score)
            dict_ = {
                save_name: save_scores
                for save_name, save_scores in zip(save_names, save_scores)
            }

        final_result = self.save_score(
            dict_,
            filename=filename,
        )
        scores = results.update(final_result)
        return Path(path, filename)

    def __call__(
        self,
    ):
        """Score the predictions from the file and updates best score."""
        path = self.files["path"] if "path" in self.files else "."
        path = Path(path, my_hash(self._asdict()))
        ground_truth = self.files.pop("ground_truth_file", None)
        predictions = self.files.pop("predictions_file", None)
        score_dict = self.files.pop("score_dict_file", None)

        predictions_file = Path(path, predictions)
        ground_truth_file = Path(path, ground_truth)
        scores_file = Path(path, score_dict)
        test = self.read_data_from_json(predictions_file)
        true = self.read_data_from_json(ground_truth_file)
        scores = self.score(true, test)
        path = self.save_results(scores=scores, path=path, filename=scores_file)
        return path.resolve()

    def __str__(self):
        string = "Scorer with scorers: "
        names = self.scorers.keys()
        scorers = [self.scorers[name]["name"] for name in names]
        for scorer, score in zip(names, scorers):
            string += "{}: {}".format(scorer, score)
        return string


if __name__ == "__main__":
    config = """
    model:
        init:
            loss: "hinge"
            name: sklearn.linear_model.SGDClassifier
        files:
            model_path : model
            model_filetype : pickle
        fit:
            epochs: 1000
            learning_rate: 1.0e-08
            log_interval: 10
    data:
        sample:
            shuffle : True
            random_state : 42
            train_size : 800
            stratify : True
        add_noise:
            train_noise : 1
            time_series : True
        name: classification
        files:
            data_path : data
            data_filetype : pickle
        generate:
            n_samples: 1000
            n_features: 2
            n_informative: 2
            n_redundant : 0
            n_classes: 2
        sklearn_pipeline:
            - sklearn.preprocessing.StandardScaler
        transform:
            sklearn.preprocessing.StandardScaler:
                with_mean : true
                with_std : true
                X_train : true
                X_test : true
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
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    yaml.add_constructor("!Scorer:", Scorer)
    scorer = yaml.load("!Scorer:\n" + str(config), Loader=yaml.Loader)
    result_path = scorer()
