import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import unittest
import os
import tempfile
import shutil
from deckard.base.generator import Generator
from pandas import DataFrame


# TODO other datasets
class testGenerator(unittest.TestCase):
    def setUp(self):
        self.filename = "test"
        self.path = tempfile.mkdtemp()
        config_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "configs"
        )
        self.config_folder = config_folder
        self.model_yml = os.path.join(self.config_folder, "model.yml")
        self.preprocess_yml = os.path.join(self.config_folder, "preprocess.yml")
        self.attack_yml = os.path.join(self.config_folder, "attack.yml")
        self.defend_yml = os.path.join(self.config_folder, "defend.yml")
        self.model_yml = os.path.join(self.config_folder, "model.yml")
        self.featurize_yml = os.path.join(self.config_folder, "featurize.yml")

    def test_init(self):
        gen = Generator(self.model_yml, self.path, "models")
        self.assertEqual(gen.path, self.path)
        self.assertEqual(gen.params_file, self.model_yml)
        self.assertEqual(gen.input, os.path.join(self.path, "configs", self.model_yml))
        self.assertEqual(gen.output, os.path.join(self.path, "models"))
        self.assertEqual(gen.params, gen.set_config())
        self.assertIsInstance(gen.list, list)

    def test_call(self):
        gen = Generator(self.model_yml, self.path, "test")
        paths = gen(self.filename)
        self.assertIsInstance(paths, DataFrame)

    def test_hash(self):
        gen = Generator(self.model_yml, self.path, "models")
        gen2 = Generator(self.model_yml, self.path, "models")
        gen3 = Generator(self.attack_yml, self.path, "models")
        self.assertEqual(hash(gen), hash(gen2))
        self.assertNotEqual(hash(gen), hash(gen3))

    def test_eq(self):
        gen = Generator(self.model_yml, self.path, "models")
        gen2 = Generator(self.model_yml, self.path, "models")
        gen3 = Generator(self.attack_yml, self.path, "models")
        self.assertEqual(gen, gen2)
        self.assertNotEqual(gen, gen3)

    def test_set_config(self):
        gen = Generator(self.model_yml, self.path, "models")
        self.assertEqual(gen.set_config(), gen.params)

    def test_generate_tuple_list_from_yml(self):
        gen = Generator(self.model_yml, self.path, "models")
        self.assertIsInstance(gen.list, list)
        self.assertIsInstance(gen.list[0], tuple)
        self.assertIsInstance(gen.list[0][0], str)
        self.assertIsInstance(gen.list[0][1], dict)

    def test_generate_json(self):
        gen = Generator(self.model_yml, self.path, "models")
        name = gen.list[0][0]
        params = gen.list[0][1]
        gen.generate_json(
            path=self.path, filename=self.filename, name=name, params=params
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.path, self.filename + ".json"))
        )

    def test_generate_yml(self):
        gen = Generator(self.model_yml, self.path, "models")
        name = gen.list[0][0]
        params = gen.list[0][1]
        gen.generate_yml(
            path=self.path, filename=self.filename, name=name, params=params
        )
        self.assertTrue(os.path.exists(os.path.join(self.path, self.filename)))

    def test_generate_directory_tree(self):
        gen = Generator(self.model_yml, self.path, "models")
        gen.generate_directory_tree("params")
        input_len = len(gen.list)
        self.assertTrue(os.path.exists(os.path.join(self.path)))
        self.assertTrue(os.path.exists(os.path.join(self.path, "models")))
        folders = os.listdir(os.path.join(self.path, "models"))
        output_len = len(folders)
        self.assertTrue(input_len == output_len)
        for folder in folders:
            self.assertTrue(os.path.exists(os.path.join(self.path, "models", folder)))
            self.assertTrue(
                os.path.exists(os.path.join(self.path, "models", folder, "configs"))
            )

    def test_generate_experiment_list(self):
        gen = Generator(self.model_yml, self.path, "list")
        df = gen.generate_experiment_list(self.filename)
        self.assertIsInstance(df, DataFrame)
        self.assertEqual(["object_type", "params", "ID"], list(df.columns))

    def tearDown(self):
        try:
            shutil.rmtree(self.path)
        except PermissionError:
            pass
