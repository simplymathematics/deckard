import unittest

import optuna

from deckard.optuna_callback import load_optuna_studies_dataframe


class TestOptunaCallbackQuery(unittest.TestCase):
    def _build_storage(self):
        storage = optuna.storages.InMemoryStorage()

        for study_name, values in {
            "alpha_one": [0.2, 0.7],
            "beta_two": [0.1, 0.9],
        }.items():
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                load_if_exists=True,
            )
            for value in values:
                trial = study.ask()
                study.tell(trial, value)
        return storage

    def test_query_multiple_studies_and_state_filters(self):
        storage = self._build_storage()
        frame = load_optuna_studies_dataframe(
            storage=storage,
            study_names=["alpha_one", "beta_two"],
            trial_states=["COMPLETE"],
        )
        self.assertGreaterEqual(len(frame), 4)
        self.assertIn("study_name", frame.columns)
        self.assertTrue(set(frame["study_name"]).issubset({"alpha_one", "beta_two"}))

    def test_query_applies_trial_range_sort_and_slice(self):
        storage = self._build_storage()
        frame = load_optuna_studies_dataframe(
            storage=storage,
            study_name="alpha_one",
            trial_number_range=(0, 1),
            sort_by="value",
            ascending=False,
            row_slice=(0, 1),
        )
        self.assertEqual(len(frame), 1)
        self.assertIn("value", frame.columns)

    def test_query_projection_columns_and_offset_limit(self):
        storage = self._build_storage()
        frame = load_optuna_studies_dataframe(
            storage=storage,
            study_name="beta_two",
            columns=["study_name", "value", "number"],
            offset=0,
            limit=1,
        )
        self.assertEqual(list(frame.columns), ["study_name", "value", "number"])
        self.assertEqual(len(frame), 1)
