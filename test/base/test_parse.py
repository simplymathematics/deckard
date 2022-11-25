# import os
# import tempfile
# import unittest
# import warnings
# import yaml
# from sklearn.model_selection import GridSearchCV
# from sklearn.tree import DecisionTreeClassifier

# from deckard.base import Data, Experiment
# from deckard.base.parse import (
#     generate_object_list_from_tuple,
#     generate_tuple_list_from_yml,
# )
# from deckard.layers.utils import generate_experiment_list

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)


# class testParse(unittest.TestCase):
#     def setUp(self):
#         self.path = tempfile.mkdtemp()
#         self.file = "test_filename"
#         config_file = os.path.join(
#             os.path.dirname(os.path.realpath(__file__)),
#             "configs",
#         )
#         self.configs = [x for x in os.listdir(config_file) if x.endswith(".yml")]
#         self.yml_lists = []
#         self.config_folder = config_file
#         self.model_yml = os.path.join(self.config_folder, "model.yml")
#         self.preprocess_yml = os.path.join(self.config_folder, "preprocess.yml")
#         self.attack_yml = os.path.join(self.config_folder, "attack.yml")
#         self.model = DecisionTreeClassifier()
#         self.pipeline = [("model", self.model)]
#         self.grid = GridSearchCV(self.model, {"max_depth": [1, 2]})
#         self.preprocessor = [("preprocess", self.grid)]

#     def test_generate_tuple_list_rom_yml(self):
#         tuple_list = generate_tuple_list_from_yml(self.model_yml)
#         self.assertIsInstance(tuple_list, list)
#         self.assertIsInstance(tuple_list[0], tuple)
#         self.assertIsInstance(tuple_list[0][0], str)
#         self.assertIsInstance(tuple_list[0][1], dict)

#     def test_generate_object_list_from_tuple(self):
#         tuple_list = generate_tuple_list_from_yml(self.model_yml)
#         obj_list = generate_object_list_from_tuple(tuple_list)
#         self.assertIsInstance(obj_list, list)
#         self.assertIsInstance(obj_list[0], object)
#         self.assertIn("base_estimator", obj_list[0].__dict__)

#     def test_generate_experiment_list(self):
#         data = yaml.load("!Data", Loader=yaml.FullLoader)
#         tuple_list = generate_tuple_list_from_yml(self.model_yml)
#         obj_list = generate_object_list_from_tuple(tuple_list)
#         experiment_list = generate_experiment_list(obj_list, data_list=data)
#         self.assertIsInstance(experiment_list, list)
#         self.assertIsInstance(experiment_list[0], Experiment)

#     def tearDown(self):
#         import shutil

#         shutil.rmtree(self.path)
