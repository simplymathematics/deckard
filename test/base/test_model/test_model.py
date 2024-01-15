import unittest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree
import os
import numpy as np
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
import tensorflow as tf

from deckard.base.model import Model

tf.config.run_functions_eagerly(True)

this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


class testModel(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
    config_file = "classification.yaml"
    file = "model.pkl"

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(), version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.model = instantiate(config=self.cfg)
        self.model = self.model
        self.dir = mkdtemp()
        self.model_file = Path(self.dir, self.file).as_posix()

    def test_init(self):
        self.assertTrue(isinstance(self.model, Model))

    def test_call(self):
        model_file = Path(self.dir, self.file).as_posix()
        predictions_file = Path(self.dir, "predictions.pkl").as_posix()
        probability_file = Path(self.dir, "probability.pkl").as_posix()
        self.model(
            model_file=model_file,
            predictions_file=predictions_file,
            probabilities_file=probability_file,
        )
        self.assertTrue(Path(model_file).exists())
        self.assertTrue(Path(predictions_file).exists())
        self.assertTrue(Path(probability_file).exists())

    def test_hash(self):
        old_hash = hash(self.model)
        self.assertIsInstance(old_hash, int)
        self.model()
        new_hash = hash(self.model)
        self.assertEqual(old_hash, new_hash)

    def test_initialize(self):
        data, model = self.model.initialize()
        self.assertIsInstance(data, list)
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))

    def tearDown(self) -> None:
        rmtree(self.dir)


class testModelMethods(testModel):
    def test_fit(self):
        data, model = self.model.initialize()
        data = [data[i][:10] for i in range(len(data))]
        model, time_dict = self.model.fit(data=data, model=model)
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue("train_time" in time_dict.keys())
        self.assertTrue("train_time_per_sample" in time_dict.keys())

    def test_predict(self):
        data, model = self.model.initialize()
        data = [data[i][:10] for i in range(len(data))]
        model, time_dict = self.model.fit(data=data, model=model)
        predictions, time_dict = self.model.predict(data=data, model=model)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertTrue("predict_time" in time_dict.keys())
        self.assertTrue("predict_time_per_sample" in time_dict.keys())

    def test_predict_proba(self):
        data, model = self.model.initialize()
        data = [data[i][:10] for i in range(len(data))]
        model, time_dict = self.model.fit(data=data, model=model)
        probabilities, time_dict = self.model.predict_proba(data=data, model=model)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertTrue("predict_proba_time" in time_dict.keys())

    def test_predict_log_proba(self):
        data, model = self.model.initialize()
        data = [data[i][:10] for i in range(len(data))]
        model, time_dict = self.model.fit(data=data, model=model)
        log_probas, time_dict = self.model.predict_log_loss(data=data, model=model)
        self.assertIsInstance(log_probas, np.ndarray)
        self.assertTrue("predict_log_proba_time" in time_dict.keys())
        self.assertTrue("predict_log_proba_time_per_sample" in time_dict.keys())

    def test_time_dict(self):
        data, model = self.model.initialize()
        data = [data[i][:10] for i in range(len(data))]
        model, time_dict = self.model.fit(data=data, model=model)
        _, new_time_dict = self.model.predict(data=data, model=model)
        time_dict.update(new_time_dict)
        self.assertTrue("train_time" in time_dict.keys())
        self.assertTrue("train_time_per_sample" in time_dict.keys())
        self.assertTrue("train_start_time" in time_dict.keys())
        self.assertTrue("train_end_time" in time_dict.keys())
        self.assertTrue("predict_time" in time_dict.keys())
        self.assertTrue("predict_time_per_sample" in time_dict.keys())
        self.assertTrue("predict_start_time" in time_dict.keys())
        self.assertTrue("predict_end_time" in time_dict.keys())
        self.assertTrue(time_dict["train_time"] > 0)
        self.assertTrue(time_dict["train_time_per_sample"] > 0)
        self.assertTrue(time_dict["predict_time"] > 0)
        self.assertTrue(time_dict["predict_time_per_sample"] > 0)


class testTorchModel(testModel):
    config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
    config_file = "torch_mnist.yaml"


class testTorchModelfromDict(testModel):
    config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
    config_file = "torch_defaults.yaml"


# class testKerasModel(testModel):
#     config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
#     config_file = "keras_mnist.yaml"


# class testTFV2Model(testModel):
#     config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
#     config_file = "tf_mnist.yaml"
#     file = "model.tf"

# class testTFV2Model(testModel):
#     config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
#     config_file = "tf_mnist.yaml"
#     file = "model.tf"
