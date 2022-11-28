import json
import logging
import os
from pathlib import Path
from .hashable import BaseHashable, my_hash
import pandas as pd
import yaml
import collections
from .utils import factory

logger = logging.getLogger(__name__)


class Scorer(
    collections.namedtuple(
        typename="Scorer",
        field_names="data, scorers, files, attack, model,  plots",
        defaults=({}, {}, {}, {}),
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

    def save_results(self, scores: dict, filename: str,) -> None:
        """
        Saves all data to specified folder, using default filenames.
        """
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
        return Path(filename).resolve()

    def __call__(
        self,
    ):
        """Score the predictions from the file and updates best score."""
        path = self.files["path"] if "path" in self.files else "."
        path = Path(path, my_hash(self._asdict()))
        path.mkdir(parents=True, exist_ok=True)
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
    from experiment import config
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    yaml.add_constructor("!Scorer:", Scorer)
    scorer = yaml.load("!Scorer:\n" + str(config), Loader=yaml.Loader)
    result_path = scorer()
