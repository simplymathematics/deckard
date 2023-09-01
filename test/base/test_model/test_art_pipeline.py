import unittest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree
import os
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from deckard.base.model.art_pipeline import ArtPipeline


this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


class testArtPipeline(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
    config_file = "classification.yaml"

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.model = instantiate(config=self.cfg)
        self.dir = mkdtemp()

    def test_init(self):
        self.assertTrue(isinstance(self.model.art, ArtPipeline))

    def test_call(self):
        data, model = self.model.initialize()
        model = self.model.art(data=data, model=model)
        self.assertTrue("art" in str(type(model)).lower())
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))

    def test_hash(self):
        old_hash = hash(self.model.art)
        self.assertIsInstance(old_hash, int)
        new_cfg = self.cfg
        new_model = instantiate(config=new_cfg)
        new_hash = hash(new_model.art)
        self.assertEqual(old_hash, new_hash)
        data, model = self.model.initialize()
        _ = new_model.art(data=data, model=model)
        hash_after_call = hash(new_model.art)
        self.assertEqual(old_hash, hash_after_call)

    def tearDown(self) -> None:
        rmtree(self.dir)


# class testArtPipeline(unittest.TestCase):
#     name: str = "initialize"
#     library: str = "sklearn-svc"
#     kwargs: dict = {}
#     config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
#     config_file = "classification.yaml"

#     def setUp(self):
#         with initialize_config_dir(
#             config_dir=Path(self.config_dir).resolve().as_posix(),
#             version_base="1.3",
#         ):
#             cfg = compose(config_name=self.config_file)
#         self.cfg = cfg
#         self.model = instantiate(config=self.cfg)
#         self.stage = self.model.art.pipeline[self.name]
#         self.dir = mkdtemp()

#     def test_init(self):
#         self.assertTrue(isinstance(self.stage, ArtPipelineStage))

#     def test_call(self):
#         model = self.model
#         model = self.model.art.pipeline[self.name](
#         )
#         print(type(model))
#         input("Press Enter to continue...")
#         self.assertTrue("art" in str(type(model)).lower())
#         self.assertTrue(hasattr(model, "fit"))
#         self.assertTrue(hasattr(model, "predict"))

#     def test_hash(self):
#         old_hash = hash(self.stage)
#         self.assertIsInstance(old_hash, int)
#         stage = self.model.art.pipeline[self.name]
#         new_hash = hash(stage)
#         self.assertEqual(old_hash, new_hash)
#         data, _ = self.model.initialize()
#         model = self.model.init()
#         _ = self.model.art(data=data, model=model)
#         hash_after_call = hash(self.model.art.pipeline[self.name])
#         self.assertEqual(old_hash, hash_after_call)

#     def tearDown(self) -> None:
#         rmtree(self.dir)


class testTorchArtPipeline(testArtPipeline):
    name: str = "initialize"
    library: str = "torch"
    kwargs: dict = {}
    config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
    config_file = "torch_mnist.yaml"


# class testKerasArtPipeline(testArtPipeline):
#     name: str = "initialize"
#     library: str = "keras"
#     kwargs: dict = {}
#     config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
#     config_file = "keras_mnist.yaml"


class testTFV2ArtPipeline(testArtPipeline):
    name: str = "initialize"
    library: str = "tfv2"
    kwargs: dict = {}
    config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
    config_file = "tf_mnist.yaml"
