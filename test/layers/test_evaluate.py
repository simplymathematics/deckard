import warnings, tempfile, unittest, os, argparse, dvc.api, yaml, subprocess
from pathlib import Path
from tqdm import tqdm
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from deckard.layers.evaluate import evaluate

class testEvaluate(unittest.TestCase):
    def setUp(self):
        self.path = os.path.abspath(tempfile.mkdtemp())
        ART_DATA_PATH = self.path
        self.file = 'test_filename'
        self.here = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.join(self.here, "..", "..", "examples", "car data"))
        self.here = os.path.realpath(os.path.join(self.here, "..", "..", "examples", "car data"))

    def test_evaluate(self):
        params = dvc.api.params_show()
        args = argparse.Namespace(**params['evaluate'])
        old_files = [path for path in Path(args.input_folder).rglob('*' + 'scores.json')]
        folders = [file.parent for file in old_files]
        evaluate(args, folder_list= folders)

        
    def test_cli(self):
        with open(os.path.join(self.here, 'dvc.yaml')) as f:
            dictionary = yaml.load(f, Loader=yaml.FullLoader)
        command = dictionary['stages']['evaluate']['cmd']
        subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    
    def tearDown(self) -> None:
        from shutil import rmtree
        rmtree(self.path)