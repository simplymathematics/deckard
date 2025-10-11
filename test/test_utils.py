import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from deckard.utils import ConfigBase, initialize_config

class DummyConfig(ConfigBase):
    a: int = 1
    b: str = "test"
    _private: str = "hidden"

    def __post_init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.a + len(self.b)

    def save_scores(self, scores, filepath):
        import json
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(scores, f)

class TestConfigBase(unittest.TestCase):

    def test_initialization(self):
        cfg = DummyConfig(a=5, b="hello")
        self.assertEqual(cfg.a, 5)
        self.assertEqual(cfg.b, "hello")
        self.assertEqual(cfg._private, "hidden")
        self.assertEqual(cfg(), 5 + len("hello"))
        with self.assertRaises(AttributeError):
            _ = cfg.non_existent

    def test_partial_initialization(self):
        cfg = DummyConfig(a=10)
        self.assertEqual(cfg.a, 10)
        self.assertEqual(cfg.b, "test")
        self.assertEqual(cfg(), 10 + len("test"))

    def test_no_arguments(self):
        cfg = DummyConfig()
        self.assertEqual(cfg.a, 1)
        self.assertEqual(cfg.b, "test")
        self.assertEqual(cfg(), 1 + len("test"))
    
    def test_initialize_config(self):
        cfg_list = []
        cfg = initialize_config(None, cfg_list, target="deckard.utils.ConfigBase")
        self.assertIsInstance(cfg, ConfigBase)
        
    def test_hash(self):
        cfg1 = DummyConfig(a=2, b="hash")
        cfg2 = DummyConfig(a=2, b="hash")
        cfg3 = DummyConfig(a=3, b="hash")
        self.assertEqual(hash(cfg1), hash(cfg2))
        self.assertNotEqual(hash(cfg1), hash(cfg3))
    
    def test_save_scores(self):
        cfg = DummyConfig(a=4, b="save")
        scores = {"accuracy": 0.95, "loss": 0.1}
        with tempfile.TemporaryDirectory() as tmpdirname:
            score_path = Path(tmpdirname) / "scores.json"
            # Use the inherited method from ConfigBase
            cfg.save_scores(scores, score_path)
            self.assertTrue(score_path.parent.exists())
            self.assertTrue(score_path.exists())

    def test_save_data(self):
        cfg = DummyConfig(a=2, b="data")
        data = pd.DataFrame(np.array([[1, 2], [3, 4]]))
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_path = Path(tmpdirname) / "data.pkl"
            cfg.save_data(data, data_path)
            self.assertTrue(data_path.exists())

    
    def test_load_data_raises_error(self):
        cfg = DummyConfig()
        with self.assertRaises(FileNotFoundError):
            cfg.load_data("non_existent_file.pkl")
    
    def test_load_data_success(self):
        cfg = DummyConfig()
        data = {"X": np.array([[5, 6], [7, 8]]), "y": np.array([1, 0])}
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_path = Path(tmpdirname) / "data.pkl"
            pd.to_pickle(data, data_path)
            loaded_data = cfg.load_data(data_path)
            self.assertTrue(np.array_equal(loaded_data["X"], data["X"]))
            self.assertTrue(np.array_equal(loaded_data["y"], data["y"]))
    
    def test_save_object(self):
        cfg = DummyConfig(a=7, b="object")
        with tempfile.TemporaryDirectory() as tmpdirname:
            obj_path = Path(tmpdirname) / "obj.pkl"
            cfg.save_object(cfg, obj_path)
            self.assertTrue(obj_path.exists())
            loaded_obj = pd.read_pickle(obj_path)
            self.assertIsInstance(loaded_obj, DummyConfig)
            self.assertEqual(loaded_obj.a, 7)
            self.assertEqual(loaded_obj.b, "object")
            self.assertEqual(loaded_obj(), 7 + len("object"))
    
    def test_save_self(self):
        cfg = DummyConfig(a=8, b="self")
        with tempfile.TemporaryDirectory() as tmpdirname:
            cfg()
            cfg.save(filepath=Path(tmpdirname) / "data.pkl")
            self.assertTrue(Path(tmpdirname, "data.pkl").exists())
            
    def test_load_self(self):
        cfg = DummyConfig(a=9, b="load")
        with tempfile.TemporaryDirectory() as tmpdirname:
            obj_path = Path(tmpdirname) / "obj.pkl"
            cfg.save(filepath=obj_path)
            loaded_cfg = cfg.load(obj_path)
            self.assertIsInstance(loaded_cfg, DummyConfig)
            self.assertEqual(loaded_cfg.a, 9)
            self.assertEqual(loaded_cfg.b, "load")
            self.assertEqual(loaded_cfg(), 9 + len("load"))

if __name__ == "__main__":
    unittest.main()
