"""Configuration for evasion attacks (adversarial examples)."""

import time
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from art.config import ART_NUMPY_DTYPE

from .base import AttackConfig, AttackTypePlugin, _AttackMixin, _sensitive_slice
from ..score.base import (
    DefaultClassifierConfig,
    DefaultRegressorConfig,
    ScorerDictConfig,
)
from ..frameworks.pytorch.torch_utils import (
    collect_subset_from_dataloader,
    is_dataloader,
    is_tensor,
    tensor_to_numpy,
)



logger = logging.getLogger(__name__)


class _EvasionAttackMixin(_AttackMixin):
    """Reusable evasion attack behavior."""

    # Declared for static analyzers
    attack_size: int
    attack_time: Union[float, None]
    attack_prediction_time: Union[float, None]
    attack_predictions: Union[object, None]
    attacked_labels: Union[object, None]
    score_dict: dict

    def __call__(
        self,
        *,
        data,
        model,
        art_model,
        attack,
        attack_type: str,
        attack_subtype: str,
    ) -> dict:
        if (attack_type or "").lower() != "evasion":
            raise ValueError(
                f"_EvasionAttackMixin received unsupported attack type: {attack_type}",
            )
        return self._evade(data, art_model, attack)

    def _evade(self, data, art_model, attack) -> dict:
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
            If True, uses the training set for evaluation; otherwie, uses the test set. Defaults to False.

        Returns
        -------
        dict
            A dictionary containing the scores and metrics of the attack evaluation.
        """
        import pandas as pd

        start_time = time.process_time()
        active_mode = self.resolve_mode_for_attack_kind("evasion")
        n, x_subset, y_subset = self.get_attack_subset(
            data,
            test=(active_mode != "train"),
        )
        x_subset = self._prepare_features_for_attack(x_subset)
        y_subset = self._prepare_labels_for_attack(y_subset)
        x_subset_art = self._prepare_features_for_art(x_subset)
        if not isinstance(y_subset, (list, np.ndarray)) and not is_tensor(y_subset):
            raise TypeError(
                f"Expected labels to be a list, numpy array, or tensor. Got {type(y_subset)}",
            )
        ben_preds = art_model.predict(x_subset_art)
        is_regression = self._is_regression_prediction_output(
            y_subset,
            ben_preds,
        )
        ben_pred_labels = self._prediction_to_labels(
            ben_preds,
            is_regression=is_regression,
        )
        if is_tensor(ben_pred_labels):
            ben_pred_labels = tensor_to_numpy(
                ben_pred_labels,
                dtype=ART_NUMPY_DTYPE,
            )
        if "AdversarialPatch" in str(type(attack)):
            # Special handling for AdversarialPatch attack
            patches = attack.generate(x=x_subset_art, y=ben_pred_labels)
            # Caclulate the scale of the patch, relative to the input size
            input_shape = x_subset_art[0].shape[
                1:
            ]  # Exclude batch dimension, channel dimension
            patch_shape = patches[0].shape[
                1:
            ]  # Exclude batch dimension, channel dimension
            # Assume that the patch is square (required by the attack)
            # Calculate the scale based on the larger input_dimension
            scale = max(
                patch_shape[0] / input_shape[0],
                patch_shape[1] / input_shape[1],
            )
            X_test_adv = attack.apply_patch(x_subset_art, scale=scale)
        else:
            X_test_adv = attack.generate(x=x_subset_art)
        end_time = time.process_time()
        self.attack_time = end_time - start_time
        logger.info(
            f"Evasion attack took {self.attack_time} seconds for {n} samples",
        )
        start_time = time.process_time()
        adv_pred = art_model.predict(X_test_adv)
        self.attack_predictions = adv_pred
        self.attacked_labels = y_subset
        # adv_pred_labels = adv_pred.argmax(axis=1)
        end_time = time.process_time()
        self.attack_prediction_time = end_time - start_time
        logger.info(
            f"Adversarial prediction took {self.attack_prediction_time} seconds for {n} samples",
        )
        adv_pred_labels = self._prediction_to_labels(
            adv_pred,
            is_regression=is_regression,
        )
        self.score_y_pred = adv_pred_labels
        self.score_y_proba = adv_pred
        y_test_numeric = self._normalize_ground_truth(
            y_subset,
            is_regression=is_regression,
        )
        if is_regression:
            benign_scorer = DefaultRegressorConfig()
            benign_scores = benign_scorer(
                y_true=y_test_numeric,
                y_pred=ben_pred_labels,
                mode=None,
            )
        else:
            full_classifier = DefaultClassifierConfig()
            label_only = {
                name: scorer
                for name, scorer in full_classifier.scorers.items()
                if not scorer.needs_proba
            }
            benign_scorer = ScorerDictConfig(scorers=label_only)
            benign_scores = benign_scorer(
                y_true=y_test_numeric,
                y_pred=ben_pred_labels,
                mode=None,
            )

        score_dict = self._score(
            attack_kind="evasion",
            y_true=y_test_numeric,
            y_pred=adv_pred_labels,
            ben_pred_labels=ben_pred_labels,
            is_classification=not is_regression,
            sensitive_features=_sensitive_slice(
                getattr(data, "_sensitive_test", None),
                n,
            ),
        )
        logger.info(
            f"Attack scoring took {self.attack_score_time} seconds for {len(adv_pred_labels)} samples and {len(self.score_dict)} scores.",
        )
        prefixed_benign = {f"benign_{k}": v for k, v in benign_scores.items()}
        self.score_dict = {**self.score_dict, **prefixed_benign, **score_dict}
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")
        self.attack = adv_pred
        return self.score_dict

    def get_attack_subset(self, data: Any, test: bool = True) -> tuple:
        """Get a subset of data for attack (supports multiple data types)."""
        import pandas as pd
        from torch.utils.data import Dataset, DataLoader, Subset

        n = self.attack_size
        if test is True:
            x_ = data.X_test
            y_ = data.y_test
        else:
            x_ = data.X_train
            y_ = data.y_train
        # Accept Subset/Dataset and convert to tensor
        if isinstance(x_, (pd.Series, np.ndarray, pd.DataFrame)) or is_tensor(x_):
            x_subset = x_[:n]
            y_subset = y_[:n]
        elif isinstance(x_, (Dataset, Subset)):
            # Convert to tensor
            loader = DataLoader(x_, batch_size=n, shuffle=False)
            batch = next(iter(loader))
            if isinstance(batch, (tuple, list)):
                x_subset = batch[0]
                y_subset = batch[1]
            else:
                x_subset = batch
                y_subset = None
        elif is_dataloader(x_):
            x_subset, y_subset = collect_subset_from_dataloader(x_, n=n)
        else:
            raise ValueError(
                f"Expected data.X_test to be a pd.Series, np.ndarray, torch Tensor, torch DataLoader, or torch Dataset/Subset. Got: {type(data.X_test)}",
            )
        # Do not flatten x_subset; preserve original shape for torch/ART models
        if y_subset is not None and is_tensor(y_subset) and y_subset.ndim > 1:
            y_subset = y_subset.view(-1)
        return n, x_subset, y_subset


@dataclass(eq=False, kw_only=True)
class EvasionAttackConfig(_EvasionAttackMixin, AttackConfig):
    """Configuration for evasion attacks that generate adversarial examples.

    Initialization params
    ---------------------
    attack_type : str
        Attack family path inherited from ``AttackConfig``. Expected family is
        ``evasion``.
    attack_params : dict[str, Any]
        Constructor kwargs forwarded to resolved ART evasion attack classes.
    init_params : dict[str, Any]
        Metadata-only declaration payload for class/type/library docs.
    plugins : list[AttackTypePlugin]
        Declarative runtime plugin specs. Default contains one
        ``AttackTypePlugin`` configured with:
        ``mixin_type: type = _EvasionAttackMixin`` and
        ``attack_type: str = 'evasion'``.

    Runtime params
    --------------
    _EvasionAttackMixin.__call__(self, *, data: Any, model: Any, art_model: Any, attack: Any, attack_type: str, attack_subtype: str) -> dict
        Runtime dispatch entrypoint invoked by ``AttackConfig.__call__``.
    _EvasionAttackMixin._evade(self, data: Any, art_model: Any, attack: Any) -> dict
        Generates adversarial examples and returns score payload.
    """

    plugins: list = field(
        default_factory=lambda: [
            AttackTypePlugin(
                mixin_type=_EvasionAttackMixin,
                attack_type="evasion",
            )
        ]
    )


