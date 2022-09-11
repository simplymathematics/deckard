import warnings, tempfile, unittest, os, argparse, dvc.api, yaml
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from deckard.layers.attack import attack

class testAttack(unittest.TestCase):
    def setUp(self):
        self.path = os.path.abspath(tempfile.mkdtemp())
        ART_DATA_PATH = self.path
        self.file = 'test_filename'
        self.here = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.join(self.here, "..", "..", "examples", "mnist"))
        self.here = os.path.realpath(os.path.join(self.here, "..", "..", "examples", "mnist"))
        
    def test_attack(self):
        params = dvc.api.params_show()
        args = argparse.Namespace(**params['attack'])
        attack(args)
    
    def test_cli(self):
        with open(os.path.join(self.here, 'dvc.yaml')) as f:
            dictionary = yaml.load(f, Loader=yaml.FullLoader)
        command = dictionary['stages']['attack']['cmd']
        output = os.popen(command).read()
    
    def tearDown(self) -> None:
        from shutil import rmtree
        rmtree(self.path)
    
    

