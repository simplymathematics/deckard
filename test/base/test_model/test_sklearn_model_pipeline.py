import unittest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree
import os
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from deckard.base.model.sklearn_pipeline import SklearnModelPipeline, SklearnModelPipelineStage


this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


class testSklearnModelPipeline(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
    config_file = "pipeline.yaml"

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
        self.assertTrue(isinstance(self.model.init.pipeline, SklearnModelPipeline))
        
    def test_call(self):
        _, model = self.model.initialize()
        self.assertTrue(hasattr(model, "steps"))

    def test_hash(self):
        old_hash = hash(self.model.init.pipeline)
        self.assertIsInstance(old_hash, int)
        new_cfg = self.cfg
        new_model = instantiate(config=new_cfg)
        new_hash = hash(new_model.init.pipeline)
        self.assertEqual(old_hash, new_hash)
        _, model = self.model.initialize()
        _ = new_model.init.pipeline(model=model)
        hash_after_call = hash(new_model.init.pipeline)
        self.assertEqual(old_hash, hash_after_call)

    def tearDown(self) -> None:
        rmtree(self.dir)


class testsSklearnModelPipelinefromDict(testSklearnModelPipeline):
    config_dir = Path(this_dir, "../../conf/model").resolve().as_posix()
    config_file = "pipeline2.yaml"