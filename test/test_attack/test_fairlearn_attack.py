import unittest
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from deckard.attack import AttackConfig
from deckard.score.attack import FairlearnAttackScorerConfig
from deckard.score.attack import FairlearnAttackScorerConfig
from deckard.score.fairness import FairlearnScoreDictConfig
from deckard.score.base import DefaultClassifierConfig
from deckard.data.fairness import FairlearnDataConfig


class TestFairlearnAttackScorer(unittest.TestCase):
    """Unit tests for FairlearnAttackScorerConfig per-group attack metrics."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_fairlearn(self):
        pytest.importorskip("fairlearn")

    def _make_data_with_sensitive(self):
        data = FairlearnDataConfig(dataset_name="adult", sensitive_columns="sex")
        data()
        return data

    def test_fairlearn_attack_scorer_instantiates(self):

        scorer = FairlearnAttackScorerConfig(evasion=DefaultClassifierConfig())
        self.assertIsInstance(scorer.evasion, FairlearnScoreDictConfig)
        self.assertIsInstance(
            scorer.membership_inference,
            FairlearnScoreDictConfig,
        )
        self.assertIsInstance(
            scorer.attribute_inference,
            FairlearnScoreDictConfig,
        )

    def test_score_evasion_with_sensitive_features_produces_group_metrics(self):
        from deckard.score.attack import FairlearnAttackScorerConfig

        scorer = FairlearnAttackScorerConfig()
        rng = np.random.default_rng(1)
        n = 20
        y_true = rng.integers(0, 2, n)
        y_pred = rng.integers(0, 2, n)
        sensitive = np.array(["a" if i % 2 == 0 else "b" for i in range(n)])
        result = scorer.score_evasion(
            ben_pred_labels=y_true,
            adv_pred_labels=y_pred,
            y_true=y_true,
            attack_size=n,
            is_classification=True,
            sensitive_features=sensitive,
        )
        group_keys = [k for k in result if "_accuracy" in k or "_f1" in k]
        self.assertTrue(
            len(group_keys) > 0,
            f"No group metrics found in {list(result)}",
        )

    def test_score_membership_with_sensitive_features_produces_group_metrics(
        self,
    ):
        from deckard.score.attack import FairlearnAttackScorerConfig

        scorer = FairlearnAttackScorerConfig()
        rng = np.random.default_rng(2)
        n = 20
        labels = rng.integers(0, 2, n)
        inferred = rng.integers(0, 2, n)
        sensitive = np.array(["a" if i % 2 == 0 else "b" for i in range(n)])
        result = scorer.score_membership(
            labels=labels,
            inferred=inferred,
            attack_size=n,
            sensitive_features=sensitive,
        )
        group_keys = [k for k in result if "membership_inference" in k]
        self.assertTrue(
            len(group_keys) > 0,
            f"No membership_inference metrics found in {list(result)}",
        )

    def test_score_attribute_with_sensitive_features_produces_group_metrics(
        self,
    ):

        scorer = FairlearnAttackScorerConfig()
        rng = np.random.default_rng(3)
        n = 20
        target = rng.integers(0, 3, n)
        inferred = rng.integers(0, 3, n)
        sensitive = np.array(["a" if i % 2 == 0 else "b" for i in range(n)])
        result = scorer.score_attribute(
            target=target,
            inferred=inferred,
            attack_size=n,
            targeted_attribute="age",
            is_classification=True,
            sensitive_features=sensitive,
        )
        group_keys = [k for k in result if "inferred_age" in k]
        self.assertTrue(
            len(group_keys) > 0,
            f"No inferred_age metrics found in {list(result)}",
        )

    def test_evasion_attack_with_fairlearn_scorer_end_to_end(self):
        from deckard.score.attack import FairlearnAttackScorerConfig

        pytest.importorskip("art")
        data = self._make_data_with_sensitive()
        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            attack_type="art.attacks.evasion.FastGradientMethod",
            attack_params={"eps": 0.1},
            attack_size=10,
            scorer=FairlearnAttackScorerConfig(),
        )
        scores = attack(data, model)
        group_keys = [k for k in scores if "_accuracy" in k or "_f1" in k]
        self.assertTrue(
            len(group_keys) > 0,
            f"Expected per-group evasion metrics, got keys: {list(scores)}",
        )

    def test_membership_inference_with_fairlearn_scorer_end_to_end(self):
        from deckard.score.attack import FairlearnAttackScorerConfig

        pytest.importorskip("art")
        data = self._make_data_with_sensitive()
        model = LogisticRegression(max_iter=200).fit(
            data.X_train.values,
            data.y_train.values,
        )
        attack = AttackConfig(
            attack_type="art.attacks.inference.membership_inference.MembershipInferenceBlackBox",
            attack_params={},
            attack_size=20,
            scorer=FairlearnAttackScorerConfig(),
        )
        scores = attack(data, model)
        group_keys = [k for k in scores if "membership_inference" in k]
        self.assertTrue(
            len(group_keys) > 0,
            f"Expected per-group membership metrics, got keys: {list(scores)}",
        )
