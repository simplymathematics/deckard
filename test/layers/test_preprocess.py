import warnings, tempfile, unittest, os, argparse, dvc.api, yaml, subprocess
from pathlib import Path
from tqdm import tqdm
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from deckard.layers.preprocess import preprocess

class testSklearnPreprocess(unittest.TestCase):
    def setUp(self):
        self.path = os.path.abspath(tempfile.mkdtemp())
        ART_DATA_PATH = self.path
        self.file = 'test_filename'
        self.here = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.join(self.here, "..", "..", "examples", "car data"))
        self.here = os.path.realpath(os.path.join(self.here, "..", "..", "examples", "car data"))

    def test_preprocess(self):
        params = dvc.api.params_show()
        args = argparse.Namespace(**params['preprocess'])
        if not os.path.exists(args.output_folder):
            os.mkdir(args.output_folder)
        ART_DATA_PATH = args.output_folder
        old_files = [path for path in Path(args.input_folder).rglob('*' + args.input_name)]
        new_folders = [str(file).replace(args.input_folder, args.output_folder).replace(args.input_name, "") for file in old_files]
        for folder in tqdm(new_folders, desc = "Adding preprocessor to each model"):
            Path(folder).mkdir(parents=True, exist_ok=True)
            assert(os.path.isdir(folder))
            preprocess(args, model_list = old_files, sub_folder = folder)

        
    def test_cli(self):
        with open(os.path.join(self.here, 'dvc.yaml')) as f:
            dictionary = yaml.load(f, Loader=yaml.FullLoader)
        command = dictionary['stages']['preprocess']['cmd']
        subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    
    def tearDown(self) -> None:
        from shutil import rmtree
        rmtree(self.path)