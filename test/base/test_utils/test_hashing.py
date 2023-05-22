import unittest
from pathlib import Path
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
from deckard.base.utils import to_dict, my_hash
import os

this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


@dataclass
class testClass:
    C: int = 1


class testFactory(unittest.TestCase):
    param_dict: dict = {"C": 1}
    ordered_dict: OrderedDict = OrderedDict({"C": 1})
    named_tuple: namedtuple = namedtuple("named_tuple", ["C"])(1)
    data_class: dataclass = testClass()
    dict_config: DictConfig = OmegaConf.create({"C": 1})

    def test_to_dict(self):
        old_dict = to_dict(self.param_dict)
        self.assertIsInstance(old_dict, dict)
        for thing in [
            "param_dict",
            "ordered_dict",
            "named_tuple",
            "data_class",
            "dict_config",
        ]:
            new_dict = to_dict(getattr(self, thing))
            self.assertIsInstance(new_dict, dict)
            self.assertDictEqual(old_dict, new_dict)

    def test_my_hash(self):
        old_hash = my_hash(self.param_dict)
        self.assertIsInstance(old_hash, str)
        for thing in [
            "param_dict",
            "ordered_dict",
            "named_tuple",
            "data_class",
            "dict_config",
        ]:
            new_hash = my_hash(getattr(self, thing))
            self.assertIsInstance(new_hash, str)
            self.assertEqual(old_hash, new_hash)
