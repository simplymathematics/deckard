import unittest
import pandas as pd
from unittest.mock import MagicMock


class TestAnjanaScorers(unittest.TestCase):
    """Tests for deckard/score/anjana.py — currently 36% coverage."""

    def _make_frame(self):
        """Return a small DataFrame with k-anonymity structure (k=2 groups of 2)."""
        return pd.DataFrame(
            {
                "age": [25, 25, 35, 35],
                "zip": ["10001", "10001", "10002", "10002"],
                "income": [50000, 60000, 70000, 80000],
                "disease": ["flu", "cold", "diabetes", "flu"],
            },
        )

    # -----------------------------------------------------------------------
    # _resolve_frame_and_context
    # -----------------------------------------------------------------------
    def test_resolve_frame_from_y_pred_dataframe(self):
        from deckard.plugins.anjana.score import _resolve_frame_and_context

        frame = self._make_frame()
        result_frame, qi, _ = _resolve_frame_and_context(
            y_pred=frame,
            quasi_ident=["age", "zip"],
        )
        self.assertIs(result_frame, frame)
        self.assertEqual(qi, ["age", "zip"])

    def test_resolve_frame_from_data_attr(self):
        from deckard.plugins.anjana.score import _resolve_frame_and_context

        frame = self._make_frame()
        data = MagicMock()
        data._X = frame
        data.quasi_identifiers = ["age", "zip"]
        data.sensitive_attribute = "disease"
        result_frame, qi, sens = _resolve_frame_and_context(data=data)
        self.assertIs(result_frame, frame)
        self.assertEqual(qi, ["age", "zip"])
        self.assertEqual(sens, "disease")

    def test_resolve_frame_raises_when_no_frame(self):
        from deckard.plugins.anjana.score import _resolve_frame_and_context

        with self.assertRaises(ValueError, msg="should require DataFrame"):
            _resolve_frame_and_context(y_pred=[1, 2, 3], quasi_ident=["age"])

    def test_resolve_frame_raises_when_no_quasi_ident(self):
        from deckard.plugins.anjana.score import _resolve_frame_and_context

        frame = self._make_frame()
        with self.assertRaises(ValueError):
            _resolve_frame_and_context(y_pred=frame, quasi_ident=[])

    def test_string_quasi_ident_coerced_to_list(self):
        from deckard.plugins.anjana.score import _resolve_frame_and_context

        frame = self._make_frame()
        _, qi, _ = _resolve_frame_and_context(y_pred=frame, quasi_ident="age")
        self.assertEqual(qi, ["age"])

    # -----------------------------------------------------------------------
    # anjana_k_anonymity_score
    # -----------------------------------------------------------------------
    def test_k_anonymity_score_via_y_pred(self):
        from deckard.plugins.anjana.score import anjana_k_anonymity_score

        frame = self._make_frame()
        score = anjana_k_anonymity_score(y_pred=frame, quasi_ident=["age", "zip"])
        self.assertGreaterEqual(score, 1.0)

    def test_k_anonymity_score_via_data_attr(self):

        from deckard.plugins.anjana.score import anjana_k_anonymity_score

        frame = self._make_frame()
        data = MagicMock()
        data._X = frame
        data.quasi_identifiers = ["age", "zip"]
        data.sensitive_attribute = "disease"
        score = anjana_k_anonymity_score(data=data)
        self.assertIsInstance(score, float)

    # -----------------------------------------------------------------------
    # anjana_l_diversity_score
    # -----------------------------------------------------------------------
    def test_l_diversity_score_requires_sens_att(self):
        from deckard.plugins.anjana.score import anjana_l_diversity_score

        frame = self._make_frame()
        with self.assertRaises(ValueError):
            anjana_l_diversity_score(y_pred=frame, quasi_ident=["age", "zip"])

    def test_l_diversity_score_succeeds(self):
        from deckard.plugins.anjana.score import anjana_l_diversity_score

        frame = self._make_frame()
        score = anjana_l_diversity_score(
            y_pred=frame,
            quasi_ident=["age", "zip"],
            sens_att="disease",
        )
        self.assertIsInstance(score, float)

    # -----------------------------------------------------------------------
    # anjana_t_closeness_score
    # -----------------------------------------------------------------------
    def test_t_closeness_score_requires_sens_att(self):
        try:
            from pycanon import anonymity as pycanon_anonymity  # noqa: F401
        except ImportError:
            self.skipTest("pycanon not installed")

        from deckard.plugins.anjana.score import anjana_t_closeness_score

        frame = self._make_frame()
        with self.assertRaises(ValueError):
            anjana_t_closeness_score(y_pred=frame, quasi_ident=["age", "zip"])

    def test_t_closeness_score_succeeds(self):
        try:
            from pycanon import anonymity as pycanon_anonymity  # noqa: F401
        except ImportError:
            self.skipTest("pycanon not installed")

        from deckard.plugins.anjana.score import anjana_t_closeness_score

        frame = self._make_frame()
        score = anjana_t_closeness_score(
            y_pred=frame,
            quasi_ident=["age", "zip"],
            sens_att="disease",
        )
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
