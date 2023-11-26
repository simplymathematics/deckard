import unittest
from pathlib import Path
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig, ListConfig
from deckard.base.utils import to_dict, my_hash
import os

this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


@dataclass
class testClass:
    C: int = 1


class testHashing(unittest.TestCase):
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
            if hasattr(self, thing):
                old_dict = to_dict(getattr(self, thing))
                self.assertIsInstance(old_dict, dict)
                new_dict = to_dict(getattr(self, thing))
                self.assertIsInstance(new_dict, dict)
                self.assertDictEqual(old_dict, new_dict)
            else:
                pass

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
            if hasattr(self, thing):
                old_hash = my_hash(getattr(self, thing))
                self.assertIsInstance(old_hash, str)
                new_hash = my_hash(getattr(self, thing))
                self.assertIsInstance(new_hash, str)
                self.assertEqual(old_hash, new_hash)


class testHashingNested(testHashing):
    param_dict: dict = {"C": 1, "D": [1, 2, 3, 4]}
    ordered_dict: OrderedDict = OrderedDict({"C": 1, "D": [1, 2, 3, 4]})
    named_tuple: namedtuple = namedtuple("named_tuple", ["C", "D"])(1, [1, 2, 3, 4])
    dict_config: DictConfig = OmegaConf.create({"C": 1, "D": [1, 2, 3, 4]})


class testHashingNestedNested(testHashing):
    D = ListConfig([1, 2, 3, 4])
    param_dict: dict = {"C": 1, "D": [1, 2, 3, 4]}
    ordered_dict: OrderedDict = OrderedDict({"C": 1, "D": D})
    named_tuple: namedtuple = namedtuple("named_tuple", ["C", "D"])(1, D)
    dict_config: DictConfig = OmegaConf.create({"C": 1, "D": D})


class testListHashing(testHashing):
    param_dict: list = [1, 2, 3, 4]
    dict_config: ListConfig = OmegaConf.create([1, 2, 3, 4])


class testStringHashing(testHashing):
    param_dict: str = "test"
    dict_config: str = "test"


class testNoneHashing(testHashing):
    param_dict: type(None) = None
    dict_config: type(None) = None


class testDataClassHashing(testHashing):
    param_dict: testClass = testClass()
    dict_config: testClass = testClass()


class testDictofNoneHashing(testHashing):
    param_dict: dict = {"C": None}
    ordered_dict: OrderedDict = OrderedDict({"C": None})
    named_tuple: namedtuple = namedtuple("named_tuple", ["C"])(None)
    data_class: dataclass = testClass(None)
    dict_config: DictConfig = OmegaConf.create({"C": None})
