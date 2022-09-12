import warnings, tempfile, unittest, os, argparse, dvc.api, yaml, subprocess
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from deckard.layers.sklearn_model import sklearn_model

class testSklearnModel(unittest.TestCase):
    def setUp(self):
        self.path = os.path.abspath(tempfile.mkdtemp())
        ART_DATA_PATH = self.path
        self.file = 'test_filename'
        self.here = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.join(self.here, "..", "..", "examples", "car data"))
        self.here = os.path.realpath(os.path.join(self.here, "..", "..", "examples", "car data"))

    def test_art_defend(self):
        params = dvc.api.params_show()
        args = argparse.Namespace(**params['train'])
        sklearn_model(args)
        
    def test_cli(self):
        with open(os.path.join(self.here, 'dvc.yaml')) as f:
            dictionary = yaml.load(f, Loader=yaml.FullLoader)
        command = dictionary['stages']['train']['cmd']
        subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    
    def tearDown(self) -> None:
        from shutil import rmtree
        rmtree(self.path)