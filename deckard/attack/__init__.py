import pandas as pd
import pickle
import time
import logging
import argparse
import warnings
import importlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from dataclasses import dataclass
from typing import Union


import numpy as np
from pathlib import Path

from ..data import DataConfig
from ..model import ModelConfig
from ..model.defend import sklearn_dict, sklearn_models
from ..utils import ConfigBase

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

supported_attacks = [
    "blackbox_membership_inference",
    "blackbox_evasion",
    "whitebox_evasion",
    "blackbox_attribute_inference",
    "whitebox_attribute_inference",
]


@dataclass
class AttackConfig(ConfigBase):
    """
    AttackConfig
    Configuration and execution class for adversarial attacks on machine learning models.
    This class provides a unified interface for configuring, executing, and scoring various types of adversarial attacks,
    including evasion, poisoning, extraction, and inference attacks. It supports integration with scikit-learn models
    and the Adversarial Robustness Toolbox (ART), and provides detailed logging and timing for attack operations.

    Attributes
    ----------

    attack_time : float, optional
        Time taken to execute the attack.
    attack_prediction_time : float, optional
        Time taken for adversarial prediction.
    attack_score_time : float, optional
        Time taken to score the attack.
    attack : object, optional
        Stores the result of the attack.
    attack_predictions : list, optional
        Stores the predictions made by the attack.
    score_dict : dict, optional
        Stores the computed scores and metrics for the attack.

    Methods
    -------
    __hash__()
        Computes a hash value for the object based on its non-private attributes.
    __post_init__()
        Initializes post-construction attributes and sets defaults.
    __call__(data, model, train=False, **kwargs)
    _get_benign_preds(data, art_model, train=False)
        Generates benign predictions and corresponding labels for a subset of data.
    _get_feature_vector_preds(data, targeted_attribute, train=False)
        Extracts a subset of feature vectors, labels, and attributes from the provided data.
    _score_attack(ben_pred_labels, adv_pred_labels, y_test_numeric)
    _evade(data, art_model, attack, train=False)
    _pop_attribute(X, targeted_attribute)
    _infer_attribute(data, art_model, attack, targeted_attribute, train=False)
        Performs an attribute inference attack on a dataset using a specified attack model and model.
    _infer_membership(data, art_model, attack, train=False)
        Performs membership inference attack on the given dataset using the specified attack and model.
    _poison()
    _extract()
    _save(filepath)

    Raises
    ------
    ValueError
        If the attack type, subtype, or model type is unsupported, or if the model is not fitted.
    NotImplementedError
        If the attack type or subtype is not implemented.
    AssertionError
        If the output scores or timing variables are not of the expected types.
    TypeError
        If the attack model's fit method does not accept the expected arguments.

    Examples
    --------
    >>> config = AttackConfig(attack_type="art.attacks.evasion.FastGradientMethod", attack_params={"eps": 0.2})
    >>> results = config(data, model)
    >>> print(results)
    """

    attack_type: str = "art.attacks.evasion.HopSkipJump"
    attack_params: dict = None
    attack_size: int = 10  # Number of samples to attack
    targeted_attribute: str = None  # For inference attacks

    def __hash__(self):
        return super().__hash__()

    def __post_init__(self):
        """
        Initializes post-construction attributes for the class.

        Sets the internal attack attribute to None. If attack_params is not provided,
        initializes it as an empty dictionary.
        """
        if self.attack_params is None:
            self.attack_params = {}
        self.attack_predictions = None
        self.attack = None
        self.score_dict = {}
        self.attack_time = None
        self.attack_prediction_time = None
        self.attack_score_time = None
        if self._target_ is None:
            self._target_ = "deckard.AttackConfig"

    def _initialize_attack(self, model, data):
        """
        Initialize an attack instance for a given model.

        This method determines the appropriate attack class and model wrapper based on the provided model and attack name.
        It validates the attack type and model compatibility, wraps the model if necessary, and instantiates the attack.
        If the attack cannot be initialized with the model (Whitebox), it falls back to a Blackbox attack.

        Parameters
        ----------
        model : object
            The model or configuration object to attack. Can be a fitted scikit-learn model or a ModelConfig instance.

        Returns
        -------
        attack : object
            The initialized attack instance.
        art_model : object
            The ART-wrapped model compatible with the attack.
        attack_type : str
            The type of attack (evasion, poisoning, extraction, inference).
        attack_subtype : str
            The subtype of the attack.

        Raises
        ------
        ValueError
            If the attack type or model type is unsupported, or if the model is not fitted.
        """
        if isinstance(model, ModelConfig):
            model = model._model
        else:
            check_is_fitted(model)
        module = importlib.import_module(self.attack_type.rsplit(".", 1)[0])
        attack_type = self.attack_type.split("attacks.")[-1].split(".")[0]
        attack_subtype = self.attack_type.split("attacks.")[-1].split(".")[1]

        # Validate attack type
        if attack_type not in ["evasion", "poisoning", "extraction", "inference"]:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        attack_class = getattr(module, self.attack_type.split(".")[-1])
        model_alias = type(model).__name__

        # Validate library support
        if model_alias in sklearn_models:
            if isinstance(model, tuple(sklearn_dict.values())):
                art_model = model
            try:
                check_is_fitted(model)
                art_model = sklearn_dict[model_alias](model)
            except NotFittedError:
                raise ValueError(f"model {model_alias} is not fitted")
        else:
            raise ValueError(f"Unsupported model type: {model_alias}")
        if self.targeted_attribute is not None and isinstance(self.targeted_attribute, str):
            feature_name = self.targeted_attribute
            index = data.X_train.columns.get_loc(feature_name)
            self.attack_params['attack_feature'] = index
        # Initialize the attack
        try:  # Assume Whitebox attack if model can be passed to the attack constructor
            attack = attack_class(art_model, **self.attack_params)
        except ValueError as e:  # If ValueError, assume Blackbox attack
            logger.warning(f"Falling back to Blackbox attack due to error: {e}")
            attack = attack_class(**self.attack_params)
        return attack, art_model, attack_type, attack_subtype

    def __call__(
        self,
        data,
        model,
        attack_file: Union[str, None] = None,
        attack_predictions_file: Union[str, None] = None,
        attack_scores_file: Union[str, None] = None,
    ):
        """
        Executes the specified attack on the provided model using the given data.

        Parameters
        ----------
        data : Any
            The input data to be used for the attack.
        model : object
            The machine learning model to be attacked.
        attack_file : str or None, optional
            File path to save the attack object. If None, the attack object is not saved. Default is None.
        attack_predictions_file : str or None, optional
            File path to save the attack predictions. If None, predictions are not saved. Default is None.
        attack_scores_file : str or None, optional
            File path to save the attack scores. If None, scores are not saved. Default is None.
        **kwargs
            Additional keyword arguments for the attack.

        Returns
        -------
        dict
            A dictionary containing attack scores and timing information.

        Raises
        ------
        ValueError
            If the attack type, subtype, or model type is unsupported, or if the model is not fitted.
        NotImplementedError
            If the attack type or subtype is not implemented.
        AssertionError
            If the output scores or timing variables are not of the expected types.
        """
        if attack_file is not None and Path(attack_file).exists():
            self.attack = self.load_object(attack_file)
        if (
            attack_predictions_file is not None
            and Path(attack_predictions_file).exists()
        ):
            self.attack_predictions = self.load_object(attack_predictions_file)
        if attack_scores_file is not None and Path(attack_scores_file).exists():
            self.score_dict = self.load_object(attack_scores_file)
        if (
            self.attack is not None
            and self.score_dict is not None
            and self.attack_predictions is not None
        ):
            required_keys = {
                "attack_success",
                "attack_time",
                "attack_prediction_time",
                "attack_score_time",
            }
            if not required_keys.issubset(self.score_dict.keys()):
                raise ValueError(
                    f"score_dict is missing required keys: {required_keys - self.score_dict.keys()}",
                )
            return self.score_dict
        elif (
            self.attack is None
        ):  # If attack is not already loaded, initialize and run it
            attack, art_model, attack_type, attack_subtype = self._initialize_attack(
                model,
                data,
            )

        # Execute the attack based on type and subtype
        if attack_type == "evasion":
            scores = self._evade(data, art_model, attack)
        elif attack_type == "poisoning":
            raise NotImplementedError("Poisoning attack not implemented yet.")
        elif attack_type == "extraction":
            raise NotImplementedError("Extraction attack not implemented yet.")
        elif attack_type == "inference":
            match attack_subtype:
                case "membership_inference":
                    scores = self._infer_membership(
                        data,
                        art_model,
                        attack,
                    )
                case "attribute_inference":
                    assert (
                        self.targeted_attribute is not None
                    ), "targeted_attribute must be specified for inference attacks"
                    targeted_attribute = self.targeted_attribute
                    scores = self._infer_attribute(
                        data,
                        art_model,
                        attack,
                        targeted_attribute=targeted_attribute,
                    )
                case _:
                    raise ValueError(
                        f"Unsupported inference attack subtype: {attack_subtype}",
                    )
        else:
            raise NotImplementedError(f"Attack type {attack_type} not implemented yet.")
        assert isinstance(scores, dict), "Scores should be a dictionary"
        assert isinstance(self.attack_time, float), "Attack time should be a float"
        assert isinstance(
            self.attack_prediction_time,
            float,
        ), "Attack prediction time should be a float"
        assert isinstance(
            self.attack_score_time,
            float,
        ), "Attack score time should be a float"
        times = {
            "attack_generation_time": self.attack_time,
            "attack_prediction_time": self.attack_prediction_time,
            "attack_score_time": self.attack_score_time,
        }
        score_dict = {**scores, **times}
        self.score_dict = score_dict

        # Save attack, predictions, and scores if file paths are provided
        if attack_file is not None:
            self.save_object(self.attack, attack_file)
        if attack_predictions_file is not None:
            self.save_object(self.attack_predictions, attack_predictions_file)
        if attack_scores_file is not None:
            self.save_object(self.score_dict, attack_scores_file)
        return score_dict

    def _get_benign_preds(self, data, art_model, train=False):
        """
        Generate benign predictions and corresponding labels for a subset of data.

        Depending on the `train` flag, selects either the training or test set, obtains predictions
        from the provided ART model, and returns the predicted labels along with the corresponding
        data subset and true labels.

        Parameters
        ----------
        data : callable
            A function that returns data splits. If `train` is True, should return
            (_, _, X_test, y_test). If `train` is False, should return (X_train, y_train, _, _).
        art_model : object
            An model object with a `predict` method that accepts numpy arrays.
        train : bool, optional
            If True, use the test set; otherwise, use the training set. Defaults to False.

        Returns
        -------
        tuple
            n (int): Number of samples in the subset (self.attack_size).
            ben_pred_labels (np.ndarray): Predicted labels for the benign samples.
            X_subset (pd.DataFrame): Subset of feature data used for prediction.
            y_subset (pd.Series or np.ndarray): True labels for the subset.
        """
        n = self.attack_size
        if train is True:
            X_train = data.X_train
            y_train = data.y_train
            X_test = data.X_test
            y_test = data.y_test
            ben_preds = art_model.predict(X_test.iloc[:n].values)
            ben_pred_labels = ben_preds.argmax(axis=1)
            X_subset = X_test.iloc[:n]
            y_subset = y_test.iloc[:n]
        else:
            X_train = data.X_train
            y_train = data.y_train
            X_test = data.X_test
            y_test = data.y_test
            ben_preds = art_model.predict(X_train.iloc[:n].values)
            ben_pred_labels = ben_preds.argmax(axis=1)
            X_subset = X_train.iloc[:n]
            y_subset = y_train.iloc[:n]
        return n, ben_pred_labels, X_subset, y_subset

    def _get_feature_vector_preds(self, data, targeted_attribute, train=False):
        """
        Extracts a subset of feature vectors, labels, and attributes from the provided data for either training or testing.

        Parameters
        ----------
        data : callable
            A function that returns tuples of (X_train, y_train, a_train, X_test, y_test, a_test) when called with targeted_attribute.
        targeted_attribute : str
            The attribute to target when extracting data.
        train : bool, optional
            If True, extracts from training data; otherwise, extracts from test data. Defaults to False.

        Returns
        -------
        tuple
            n (int): The number of samples to extract (self.attack_size).
            X_subset (pd.DataFrame or pd.Series): Subset of feature vectors.
            y_subset (pd.Series): Subset of labels.
            a_subset (pd.Series): Subset of attributes.

        Raises
        ------
        AssertionError
            If the lengths of the extracted feature vectors, labels, and attributes do not match.
        """
        n = self.attack_size
        if train is False:
            X_train = data.X_train
            y_train = data.y_train
            a_train = data.X_train[targeted_attribute]
            X_test = data.X_test
            y_test = data.y_test
            a_test = data.X_test[targeted_attribute]
            X_train = X_train.drop(columns=[targeted_attribute])
            X_test = X_test.drop(columns=[targeted_attribute])
            assert (
                len(X_test) == len(y_test) == len(a_test)
            ), "X_test, y_test, and a_test must have the same length, but got lengths: {}, {}, {}".format(
                len(X_test),
                len(y_test),
                len(a_test),
            )
            X_subset = X_test.iloc[:n]
            y_subset = y_test.iloc[:n]
            a_subset = a_test.iloc[:n]
        else:

            assert (
                len(X_train) == len(y_train) == len(a_train)
            ), "X_train, y_train, and a_train must have the same length, but got lengths: {}, {}, {}".format(
                len(X_train),
                len(y_train),
                len(a_train),
            )
            X_subset = X_train.iloc[:n]
            y_subset = y_train.iloc[:n]
            a_subset = a_train.iloc[:n]
        return n, X_subset, y_subset, a_subset

    def _score_attack(self, ben_pred_labels, adv_pred_labels, y_test_numeric):
        """
        Computes and logs various performance metrics for adversarial attack predictions.

        Parameters
        ----------
        ben_pred_labels : array-like
            Predicted labels from the benign (original) model.
        adv_pred_labels : array-like
            Predicted labels from the adversarially perturbed model.
        y_test_numeric : array-like
            True labels for the test set.

        Calculates the following metrics for the adversarial predictions:
            - Accuracy
            - Precision
            - Recall
            - F1-score
            - Success rate (agreement between benign and adversarial predictions)

        Returns
        -------
        None
            The function updates the instance's score_dict attribute with the computed metrics.
        """
        start_time = time.process_time()
        adv_accuracy = accuracy_score(y_test_numeric, adv_pred_labels)
        adv_precision = precision_score(
            y_test_numeric,
            adv_pred_labels,
            zero_division=0,
            average="weighted",
        )
        adv_recall = recall_score(
            y_test_numeric,
            adv_pred_labels,
            zero_division=0,
            average="weighted",
        )
        adv_f1 = f1_score(
            y_test_numeric,
            adv_pred_labels,
            zero_division=0,
            average="weighted",
        )
        adv_success = 1 - accuracy_score(ben_pred_labels, adv_pred_labels)
        end_time = time.process_time()
        self.attack_score_time = end_time - start_time
        self.score_dict = {
            "adversarial_accuracy": adv_accuracy,
            "adversarial_precision": adv_precision,
            "adversarial_recall": adv_recall,
            "adversarial_f1-score": adv_f1,
            "adversarial_success_rate": adv_success,
        }
        logger.info(
            f"Attack scoring took {self.attack_score_time} seconds for {len(adv_pred_labels)} samples and {len(self.score_dict)} scores.",
        )
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")

    def _evade(self, data, art_model, attack):
        """
        Executes an evasion attack on a given dataset using the specified ART model and attack method.

        This method assumes a classification task and generates adversarial examples from a subset of the test data.
        It measures and logs the time taken for both the attack generation and adversarial prediction steps.
        The method then evaluates the attack by comparing benign and adversarial predictions against the true labels,
        and stores the attack results and scores.

        Parameters
        ----------
        data : object
            The dataset containing features and labels.
        art_model : object
            The adversarial robustness toolbox (ART) model used for predictions.
        attack : object
            The ART attack object used to generate adversarial examples.
        train : bool, optional
            If True, uses the training set for evaluation; otherwise, uses the test set. Defaults to False.

        Returns
        -------
        dict
            A dictionary containing the scores and metrics of the attack evaluation.
        """
        n, ben_pred_labels, X_test_subset, y_test_subset = self._get_benign_preds(
            data,
            art_model,
        )
        start_time = time.process_time()
        X_test_adv = attack.generate(x=X_test_subset.values)
        end_time = time.process_time()
        self.attack_time = end_time - start_time
        logger.info(f"Evasion attack took {self.attack_time} seconds for {n} samples")
        start_time = time.process_time()
        adv_pred = art_model.predict(X_test_adv)
        self.predictions = adv_pred
        adv_pred_labels = adv_pred.argmax(axis=1)
        end_time = time.process_time()
        self.attack_prediction_time = end_time - start_time
        logger.info(
            f"Adversarial prediction took {self.attack_prediction_time} seconds for {n} samples",
        )
        y_test_numeric = y_test_subset.astype("category").cat.codes
        self._score_attack(ben_pred_labels, adv_pred_labels, y_test_numeric)
        self.attack = adv_pred
        return self.score_dict

    def _pop_attribute(self, X: pd.DataFrame, targeted_attribute: str) -> pd.DataFrame:
        """
        Removes the specified attribute (column) from the given DataFrame and returns the remaining data and the removed column.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the data.
        targeted_attribute : str
            The name of the attribute (column) to remove from the DataFrame.

        Returns
        -------
        Tuple[np.ndarray, pd.Series]
            A tuple containing:
                - The DataFrame values as a NumPy array with the targeted attribute removed.
                - The removed attribute as a pandas Series.

        Raises
        ------
        AssertionError
            If the targeted attribute is not found in the DataFrame columns.
        """
        assert (
            targeted_attribute in X.columns
        ), f"Targeted attribute {targeted_attribute} not found in data columns"
        X = X.copy()
        target = X.pop(targeted_attribute)
        X = X.values
        return X, target

    def _infer_attribute(
        self,
        data,
        art_model,
        attack,
        targeted_attribute,
    ):
        """
        Perform an attribute inference attack on a dataset using a specified attack model and model.

        This method fits the attack model to the provided data, performs predictions, and evaluates the
        attack's performance in inferring the targeted attribute. It logs timing and scoring information
        throughout the process.

        Parameters
        ----------
        data : object
            An object containing training and test data with attributes `X_train`, `y_train`, `_X_test`, and `_y_test`.
        art_model : object
            The model used for predictions, expected to have a `predict` method.
        attack : object
            The attack model, expected to have `fit` and `infer` methods.
        targeted_attribute : str
            The name of the attribute to be inferred.
        train : bool, optional
            If True, use training data for the attack; otherwise, use test data. Defaults to False.

        Returns
        -------
        dict
            A dictionary containing accuracy, precision, recall, and F1 score for the inferred attribute.

        Raises
        ------
        AssertionError
            If required data attributes are missing or if the test set size does not match `attack_size`.
        TypeError
            If the attack model's `fit` method does not accept the expected arguments.
        """
        assert hasattr(data, "X_train") and hasattr(
            data,
            "y_train",
        ), "DataConfig must have X_train, y_train attributes. Please ensure data() has been called."
        X_train = data.X_train.copy().values
        X_test = data.X_test.copy().values
        y_test = data.y_test.copy().values
        X_test_subset_without_feature, target = self._pop_attribute(
            pd.DataFrame(X_test, columns=data.X_test.columns),
            targeted_attribute,
        )
        target = target.tolist()
        start_time = time.process_time()
        try:
            attack.fit(x=X_test)
        except TypeError as e:
            raise e
        attack_time = time.process_time() - start_time
        logger.info(
            f"Attribute inference attack training took {attack_time} seconds for {self.attack_size} samples",
        )
        self.attack_time = attack_time
        
        preds = np.array([np.argmax(arr) for arr in art_model.predict(X_test)]).reshape(-1,1)
        unique, counts = np.unique(preds, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"Class {u}: {c} samples")
        possible_values = list(np.unique(target))
        logger.info(
            f"Possible values for targeted attribute '{targeted_attribute}': {possible_values}",
        )
        self.predictions = preds
        start_time = time.process_time()
        inferred = attack.infer(
            x=X_test_subset_without_feature,
            pred=preds,
            values=possible_values,
        )
        end_time = time.process_time()
        self.attack_prediction_time = end_time - start_time
        logger.info(
            f"Attribute inference attack scoring took {self.attack_score_time} seconds for {self.attack_size} samples",
        )
        start_time = time.process_time()
        
        # Round inferred values if they are continuous
        if any(isinstance(i, float) for i in inferred):
            inferred = [round(i) for i in inferred]
        inferred_accuracy = accuracy_score(target, inferred)
        inferred_precision = precision_score(
            target,
            inferred,
            zero_division=0,
            average="weighted",
        )
        inferred_recall = recall_score(
            target,
            inferred,
            zero_division=0,
            average="weighted",
        )
        inferred_f1 = f1_score(target, inferred, zero_division=0, average="weighted")
        end_time = time.process_time()
        self.attack_score_time = end_time - start_time
        self.score_dict = {
            f"inferred_{targeted_attribute}_accuracy": inferred_accuracy,
            f"inferred_{targeted_attribute}_precision": inferred_precision,
            f"inferred_{targeted_attribute}_recall": inferred_recall,
            f"inferred_{targeted_attribute}_f1": inferred_f1,
        }
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")
        self.attack = inferred
        return self.score_dict

    def _infer_membership(self, data, art_model, attack, train=False):
        """
        Perform membership inference attack on the given dataset using the specified attack and model.

        This method fits the attack model using training and test data, obtains benign predictions,
        performs membership inference, and scores the attack's performance.

        Parameters
        ----------
        data : object
            An object containing training and test data attributes (X_train, y_train, _X_test, _y_test).
        art_model : object
            The model/model used for benign predictions.
        attack : object
            The membership inference attack object with fit and infer methods.
        train : bool, optional
            If True, uses training mode for benign predictions. Defaults to False.

        Returns
        -------
        dict
            A dictionary containing the scores and metrics of the membership inference attack.

        Raises
        ------
        Exception
            If the attack fitting process fails.
        ValueError
            If the inferred membership type is unsupported or its length does not match the number of samples.
        """
        n, ben_pred_labels, X_test_subset, y_test_subset = self._get_benign_preds(
            data,
            art_model,
            train=train,
        )
        start_time = time.process_time()
        try:
            attack.fit(
                x=data.X_train.values,
                y=data.y_train.values,
                test_x=data.X_test.values,
                test_y=data.y_test.values,
            )
        except Exception as e:
            raise e
        end_time = time.process_time()
        self.attack_time = time.process_time() - start_time
        
        logger.info(
            f"Membership inference attack training took {self.attack_time} seconds for {n} samples",
        )
        start_time = time.process_time()
        inferred = attack.infer(x=X_test_subset.values, y=y_test_subset.values)
        end_time = time.process_time()
        self.attack_time = end_time - start_time
        logger.info(
            f"Membership inference attack took {self.attack_time} seconds for {n} samples",
        )
        if isinstance(inferred, (list, np.ndarray)):
            inferred = np.array(inferred)
        elif isinstance(inferred, pd.Series):
            inferred = inferred.values
        else:
            raise ValueError(f"Unsupported inferred type: {type(inferred)}")
        assert (
            len(inferred) == n
        ), f"Length of inferred {len(inferred)} does not match number of samples {n}"
        start_time = time.process_time()
        inferred = inferred.astype(int)
        logger.info(
            f"Membership inference prediction took {self.attack_prediction_time} seconds for {n} samples",
        )
        self.predictions = inferred
        y_test_numeric = y_test_subset.astype("category").cat.codes
        end_time = time.process_time()
        self.attack_prediction_time = end_time - start_time
        logger.info(
            f"Membership inference attack prediction took {self.attack_prediction_time} seconds for {n} samples",
        )
        start_time = time.process_time()
        self._score_attack(ben_pred_labels, inferred, y_test_numeric)
        end_time = time.process_time()
        self.attack_score_time = end_time - start_time
        logger.info(
            f"Membership inference attack scoring took {self.attack_score_time} seconds for {n} samples",
        )
        self.attack = inferred
        return self.score_dict

    def _poison(self):
        """
        Not implemented yet.
        """
        raise NotImplementedError("Poisoning attack not implemented yet.")

    def _extract(self):
        """
        Not implemented yet.
        """
        raise NotImplementedError("Extraction attack not implemented yet.")

    def _save(self, filepath: Union[str, Path]):
        """
        Saves the current object to a pickle file.

        Parameters
        ----------
        filepath : Union[str, Path]
            The path where the object should be saved.
            If the provided path does not end with '.pkl', the extension will be appended automatically.

        Side Effects
        -----------
        Serializes the object and writes it to the specified file in binary format.
        Logs an info message indicating the save location.
        """
        if not filepath.endswith(".pkl"):
            filepath += ".pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"AttackConfig saved to {filepath}")
