import unittest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree
import os
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from deckard.base.attack import Attack, AttackInitializer


this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


class testAttackInitializer(unittest.TestCase):

    config_dir = Path(this_dir, "../../conf/attack").resolve().as_posix()
    config_file = "evasion.yaml"
    file = "attack.pkl"

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.attack = instantiate(config=self.cfg)
        self.dir = mkdtemp()
        self.attack_file = Path(self.dir, self.file).as_posix()

    def test_init(self):
        self.assertTrue(isinstance(self.attack.init, AttackInitializer))

    def test_call(self):
        data, model = self.attack.model.initialize()
        obj = self.attack.init(data=data, model=model)
        self.assertTrue("art.attacks." in str(type(obj)).lower())

    def test_hash(self):
        old_hash = hash(self.attack)
        new_attack = instantiate(config=self.cfg)
        new_hash = hash(new_attack)
        self.assertEqual(old_hash, new_hash)

    def tearDown(self) -> None:
        rmtree(self.dir)


class testPoisoningAttackInitializer(testAttackInitializer):

    config_dir = Path(this_dir, "../../conf/attack").resolve().as_posix()
    config_file = "poisoning.yaml"
    file = "attack.pkl"


class testInferenceAttackInitializer(testAttackInitializer):

    config_dir = Path(this_dir, "../../conf/attack").resolve().as_posix()
    config_file = "inference.yaml"
    file = "attack.pkl"


class testExtractionAttackInitializer(testAttackInitializer):

    config_dir = Path(this_dir, "../../conf/attack").resolve().as_posix()
    config_file = "extraction.yaml"
    file = "attack.pkl"


class testAttack(unittest.TestCase):

    config_dir = Path(this_dir, "../../conf/attack").resolve().as_posix()
    config_file = "evasion.yaml"
    file = "attack.pkl"

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.attack = instantiate(config=self.cfg)
        self.dir = mkdtemp()
        self.attack_file = Path(self.dir, self.file).as_posix()

    def test_init(self):
        self.assertTrue(isinstance(self.attack, Attack))

    def test_call(self):
        attack_file = Path(self.dir, self.file).as_posix()
        adv_probabilities_file = Path(self.dir, "adv_probabilities.pkl").as_posix()
        adv_losses_file = Path(self.dir, "adv_losses.pkl").as_posix()
        adv_predictions_file = Path(self.dir, "adv_predictions.pkl").as_posix()
        result = self.attack(
            attack_file=attack_file,
            adv_probabilities_file=adv_probabilities_file,
            adv_losses_file=adv_losses_file,
            adv_predictions_file=adv_predictions_file,
        )
        time_dict = result["time_dict"]
        self.assertTrue(Path(attack_file).exists())
        self.assertTrue(Path(adv_probabilities_file).exists())
        self.assertTrue(Path(adv_losses_file).exists())
        self.assertTrue(Path(adv_predictions_file).exists())
        self.assertTrue("adv_fit_time" in time_dict)
        self.assertTrue("adv_fit_time_per_sample" in time_dict)
        self.assertTrue(
            "adv_samples" in result
            or "adv_model" in result
            or "adv_predictions" in result,
        )

    def test_hash(self):
        old_hash = hash(self.attack)
        self.assertIsInstance(old_hash, int)
        new_hash = hash(instantiate(config=self.cfg))
        self.assertEqual(old_hash, new_hash)
        self.attack()
        after = hash(self.attack)
        self.assertEqual(old_hash, after)

    def tearDown(self) -> None:
        rmtree(self.dir)


# class testPoisoningAttack(testAttack):
# TODO: Fix this class
#     config_dir = Path(this_dir, "../../conf/attack").resolve().as_posix()
#     config_file = "poisoning.yaml"
#     file = "attack.pkl"


class testInferenceAttack(testAttack):

    config_dir = Path(this_dir, "../../conf/attack").resolve().as_posix()
    config_file = "inference.yaml"
    file = "attack.pkl"


class testExtractionAttack(testAttack):

    config_dir = Path(this_dir, "../../conf/attack").resolve().as_posix()
    config_file = "extraction.yaml"
    file = "attack.pkl"
