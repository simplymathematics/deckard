import unittest
from pathlib import Path
from hydra.utils import call
import os
from deckard.base.utils import make_grid, flatten_dict, unflatten_dict

this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


class testFactory(unittest.TestCase):
    name: str = "sklearn.linear_model.LogisticRegression"
    params: dict = {"C": 1}
    grid: dict = {"C": [1, 2, 3]}
    param_dict: dict = {"model": {"init": {"name": name, "params": params}}}
    param_list: list = [{"model.init.name": name, "model.init.params": params}]

    def test_call(self):
        param_dict = {"_target_": self.name, **self.params}
        self.assertTrue(hasattr(call(param_dict), "fit"))

    def test_make_grid(self):
        grid = make_grid(self.grid)
        self.assertIsInstance(grid, list)
        self.assertIsInstance(grid[0], dict)
        self.assertEqual(len(grid), 3)

    def test_flatten_dict(self):
        flat_dict = flatten_dict(self.param_dict)
        self.assertIsInstance(flat_dict, dict)
        self.assertEqual(len(flat_dict), 2)

    def test_unflatten_dict(self):
        flat_dict = flatten_dict(self.param_dict)
        unflat_dict = unflatten_dict(flat_dict)
        self.assertIsInstance(unflat_dict, dict)
        self.assertEqual(len(unflat_dict), 1)
        self.assertEqual(len(unflat_dict["model"]["init"]), 2)
