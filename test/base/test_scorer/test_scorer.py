import unittest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree
import pickle
import numpy as np
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from art.utils import to_categorical
from deckard.base.scorer import ScorerConfig, ScorerDict
import os
import json

this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


class testScorerDict(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/scorers").resolve().as_posix()
    config_file = "default.yaml"
    score_dict_type = ".json"
    score_dict_file = "score_dict"

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(), version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = OmegaConf.to_container(cfg, resolve=True)
        self.scorers = instantiate(config=self.cfg)
        self.directory = mkdtemp()
        self.file = Path(
            self.directory, self.score_dict_file + self.score_dict_type,
        ).as_posix()
        true_file = "true.pkl"
        preds_file = "preds.pkl"
        self.preds_file = Path(self.directory, preds_file).as_posix()
        self.true_file = Path(self.directory, true_file).as_posix()

    def test_init(self):
        self.assertIsInstance(self.scorers, ScorerDict)

    def test_hash(self):
        old_hash = hash(self.scorers)
        y_pred = np.random.randint(0, 2, size=(100,))
        y_true = np.random.randint(0, 2, size=(100,))
        self.assertIsInstance(old_hash, int)
        self.scorers(y_pred, y_true)
        new_hash = hash(self.scorers)
        self.assertEqual(old_hash, new_hash)

    def test_iter(self):
        for scorer in self.scorers:
            self.assertIsInstance(scorer[0], str)
            self.assertIsInstance(scorer[1], ScorerConfig)

    def test_len(self):
        self.cfg.pop("_target_", None)
        self.assertEqual(len(self.scorers), len(self.cfg))

    def test_getitem(self):
        scorer = self.scorers["accuracy"]
        self.assertIsInstance(scorer, ScorerConfig)
        self.assertEqual(scorer.alias, "accuracy_score")
        self.assertEqual(scorer.name, "sklearn.metrics.accuracy_score")
        self.assertEqual(scorer.direction, "maximize")
        self.assertEqual(scorer.args, ["y_true", "y_pred"])

    def test_call(self):
        y_pred = np.random.randint(0, 2, size=(100,))
        y_true = np.random.randint(0, 2, size=(100,))
        y_pred = to_categorical(y_pred)
        y_true = to_categorical(y_true)
        with open(self.preds_file, "wb") as f:
            pickle.dump(y_pred, f)
        with open(self.true_file, "wb") as f:
            pickle.dump(y_true, f)
        score_dict = {}
        with open(Path(self.file).with_suffix(".json"), "w") as f:
            json.dump(score_dict, f)
        score_dict = self.scorers(
            y_pred,
            y_true,
            score_dict_file=self.file,
            labels_file=self.true_file,
            predictions_file=self.preds_file,
        )
        for scorer in self.scorers:
            self.assertTrue(scorer[0] in score_dict)
        self.assertTrue(Path(self.file).exists())

    def test_save(self):
        score_dict = {"test": 1}
        self.scorers.save(score_dict, self.file)
        self.assertTrue(Path(self.file).exists())

    def test_load(self):
        score_dict = {"test": 1}
        self.scorers.save(score_dict, self.file)
        score_dict = self.scorers.load(self.file)
        self.assertIsInstance(score_dict, dict)
        self.assertTrue("test" in score_dict)
        self.assertEqual(score_dict["test"], 1)

    def tearDown(self) -> None:
        rmtree(self.directory)


class testScorerConfig(unittest.TestCase):
    name: str = "sklearn.metrics.accuracy_score"
    alias: str = "accuracy"
    params: dict = {}
    args: list = ["y_pred", "y_true"]
    direction: str = "maximize"
    directory: str = None

    def setUp(self):
        true_file = "true.pkl"
        preds_file = "preds.pkl"
        self.scorer = ScorerConfig(
            name=self.name,
            alias=self.alias,
            params=self.params,
            direction=self.direction,
            args=self.args,
        )
        temp = mkdtemp()
        self.directory = temp if self.directory is None else self.directory
        self.preds_file = Path(self.directory, preds_file).as_posix()
        self.true_file = Path(self.directory, true_file).as_posix()

    def test_init(self):
        self.assertIsInstance(self.scorer, ScorerConfig)

    def test_hash(self):
        old_hash = hash(self.scorer)
        self.assertIsInstance(old_hash, int)
        new_scorer = ScorerConfig(
            name=self.name,
            alias=self.alias,
            params=self.params,
            direction=self.direction,
            args=self.args,
        )
        new_hash = hash(new_scorer)
        self.assertEqual(old_hash, new_hash)

    def test_score(self):
        y_pred = np.random.randint(0, 2, size=(100,))
        y_pred = to_categorical(y_pred)
        y_true = np.random.randint(0, 2, size=(100,))
        y_true = to_categorical(y_true)
        score = self.scorer.score(y_pred, y_true)
        if isinstance(score, float):
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
        elif isinstance(score, (list, tuple)):
            for s in score:
                self.assertIsInstance(s, float)
                self.assertGreaterEqual(s, 0)
                self.assertLessEqual(s, 1)
        else:
            raise ValueError("Score must be either a float or a list/tuple of floats")

    def test_call(self):
        y_pred = np.random.randint(0, 2, size=(100,))
        y_pred = to_categorical(y_pred)
        y_true = np.random.randint(0, 2, size=(100,))
        y_true = to_categorical(y_true)
        with open(self.preds_file, "wb") as f:
            pickle.dump(y_pred, f)
        with open(self.true_file, "wb") as f:
            pickle.dump(y_true, f)
        score = self.scorer(y_pred, y_true)
        if isinstance(score, float):
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
        elif isinstance(score, (list, tuple)):
            for s in score:
                self.assertIsInstance(s, float)
                self.assertGreaterEqual(s, 0)
                self.assertLessEqual(s, 1)
        else:
            raise ValueError("Score must be either a float or a list/tuple of floats")

    def tearDown(self) -> None:
        rmtree(self.directory)


class testComputeAccuracy(testScorerConfig):
    name: str = "art.utils.compute_accuracy"
    alias: str = "compute_accuracy"
    params: dict = {"abstain": True, "args": ["y_true", "y_pred"]}
    direction: str = "maximize"


model_config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
model_config_file = "classification.yaml"
with initialize_config_dir(
    config_dir=Path(model_config_dir).resolve().as_posix(), version_base="1.3",
):
    cfg = compose(config_name=model_config_file)
model_cfg = cfg


class testScorerDictWithArgs(testScorerDict):
    config_dir = Path(this_dir, "../../conf/scorers").resolve().as_posix()
    config_file = "args.yaml"
    score_dict_type = ".pkl"
    score_dict_file = "score_dict"


class testScorerDictfromDict(testScorerDict):
    def setUp(self):
        self.cfg = {
            "accuracy": {
                "name": "sklearn.metrics.accuracy_score",
                "alias": "accuracy_score",
                "args": ["y_true", "y_pred"],
                "params": {},
                "direction": "maximize",
            },
            "log_loss": {
                "name": "sklearn.metrics.log_loss",
                "alias": "log_loss",
                "args": ["y_true", "y_pred"],
                "params": {},
                "direction": "minimize",
            },
        }
        self.scorers = ScorerDict(**self.cfg)
        self.directory = mkdtemp()
        self.file = Path(
            self.directory, self.score_dict_file + self.score_dict_type,
        ).as_posix()
        true_file = "true.pkl"
        preds_file = "preds.pkl"
        self.preds_file = Path(self.directory, preds_file).as_posix()
        self.true_file = Path(self.directory, true_file).as_posix()


# class testScorerDictErrors(testScorerDict):
#     config_dir = Path(this_dir, "../../conf/scorers").resolve().as_posix()
#     config_file = "errors.yaml"
#     score_dict_type = ".pkl"
#     score_dict_file = "score_dict"
