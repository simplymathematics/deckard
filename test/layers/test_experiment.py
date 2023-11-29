import unittest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree, copyfile
import os
import yaml
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from deckard.layers.utils import save_params_file
from deckard.layers.experiment import get_dvc_stage_params, run_stage, get_stages, run_stages



this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()
# this_dir = Path.cwd()


class testGetDVCStageParams(unittest.TestCase):
    config_dir = Path(this_dir, "../conf/experiment").resolve().as_posix()
    dvc_repository = Path(this_dir,"../pipelines/evasion/").resolve().as_posix()
    config_file = "evasion.yaml"
    params_file = "params.yaml"
    pipeline_file = "dvc.yaml"
    stage = "train"
    dir = mkdtemp()
    

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.cfg["files"]["directory"] = self.dir
        self.exp = instantiate(config=self.cfg)
        save_params_file(
            config_dir=self.config_dir,
            config_file=self.config_file,
            params_file=Path(self.dvc_repository, self.params_file).as_posix(),
            overrides=[],
        )
        
    def testGetDVCStageParams(self):
        old_cw = os.getcwd()
        os.chdir(self.dvc_repository)
        print(os.listdir(self.dvc_repository))
        print(Path(".").resolve().as_posix())
        input("Press Enter to continue...")
        params = get_dvc_stage_params(
            stage=self.stage,
            params_file=self.params_file,
            pipeline_file=self.pipeline_file,
            directory=".",
        )
        self.assertTrue(isinstance(params, dict))
        with open(self.pipeline_file, "r") as f:
            listed_params = yaml.safe_load(f)['stages'][self.stage]['params']
        parsed_params = list(params.keys())
        print(f"parsed_params: {parsed_params}")
        print(f"listed_params: {listed_params}")
        os.chdir(old_cw)
        os.listdir(self.dvc_repository)
        input("Press Enter to continue...")
        
        parsed_params.sort()
        listed_params.sort()
        self.assertEqual(parsed_params, listed_params)

    
    
        
    def tearDown(self) -> None:
        rmtree(self.params_file)
        rmtree(self.dir)