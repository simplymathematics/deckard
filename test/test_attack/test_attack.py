import unittest
from pathlib import Path
import os
import tempfile
import shutil
from deckard.data import DataConfig
from deckard.model.defend import DefenseConfig
from deckard.attack import AttackConfig


class TestAttackConfig(unittest.TestCase):
    def setUp(self):
        self.attack_params = {"max_iter": 10, "init_eval": 5, "max_eval": 20}
        self.attack_type = "art.attacks.evasion.HopSkipJump"
        self.attack = AttackConfig(
            attack_type=self.attack_type,
            attack_params=self.attack_params,
        )
        self.tmpdir = tempfile.mkdtemp()
        self.attack_file = os.path.join(self.tmpdir, "attack.pkl")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_post_init(self):
        self.assertTrue(hasattr(self.attack, "attack_type"))
        self.assertTrue(hasattr(self.attack, "attack_params"))

    def test_save_and_load_attack(self):
        self.attack.save(self.attack_file)
        self.assertTrue(Path(self.attack_file).exists())
        loaded_attack = AttackConfig()
        loaded_attack.load(self.attack_file)
        self.assertEqual(loaded_attack.attack_type, self.attack.attack_type)
        self.assertEqual(loaded_attack.attack_params, self.attack.attack_params)

    def test_attack_metrics(self):
        # Mock data for testing
        ben_pred_labels = [0, 1, 0]
        adv_pred_labels = [0, 0, 0]
        y_test_numeric = [0, 1, 0]
        self.attack._score_attack(ben_pred_labels, adv_pred_labels, y_test_numeric)
        metrics = self.attack.score_dict
        self.assertIn("evasion_success_rate", metrics)

    def test_call_attack(self):
        # Mock data for testing
        data = DataConfig()
        data()
        model = DefenseConfig()
        model(data=data)
        result = self.attack(data, model)
        self.assertIsNotNone(result)
        self.assertIn("evasion_success_rate", result)


if __name__ == "__main__":
    unittest.main()
