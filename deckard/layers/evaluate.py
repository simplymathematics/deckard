import os, logging, argparse
from pathlib import Path
from tqdm import tqdm
import dvc.api
from typing import Union
from deckard.base import Data, Scorer, generate_experiment_list, generate_object_list_from_tuple, generate_tuple_list_from_yml

logger = logging.getLogger(__name__)

def evaluate(args, folder_list:Union[str,Path]) -> None:    
    big_dict = {}
    for sub_folder in tqdm(folder_list, desc='Evaluating'):
        sub_folder = sub_folder.resolve()
        try:
            scorer = Scorer(config = args.config)
            ground_truth_file = sub_folder.joinpath(args.ground_truth_file)
            predictions_file = sub_folder.joinpath(args.predictions_file)
            assert os.path.exists(ground_truth_file), f"Ground truth file {ground_truth_file} does not exist."
            assert os.path.exists(predictions_file), f"Predictions file {predictions_file} does not exist."
            scorer = scorer(ground_truth_file = args.ground_truth_file, predictions_file = args.predictions_file, path = sub_folder)
            scores_file = os.path.join(sub_folder, 'scores.json')
            exist = os.listdir(sub_folder)
            inside = os.path.dirname(sub_folder)
            assert os.path.exists(scores_file), f"Scores file {scores_file} does not  {inside} but {exist} exist."
            big_dict[sub_folder] = scorer.scores
            
        except Exception as e:
            logger.warning(f"Could not evaluate {sub_folder} because {e}")
            continue
        
        
if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description='Run a preprocessor on a dataset')
    parser.add_argument('--input_folder', '-i', type=str, default = ".", help='Path to the model')
    parser.add_argument('--ground_truth_file', '-true', type=str, default = "ground_truth.csv", help='Path to the model')    
    parser.add_argument('--predictions_file', '-pred', type=str, default = "predictions.csv", help='Path to the model')
    parser.add_argument('--layer_name','-l', type=str, required = True, help='Name of layer, e.g. "attack"')
    parser.add_argument('--config','-c', type=str, default = None, help='Path to the attack config file')
    args = parser.parse_args()
    # parse arguments
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    args = argparse.Namespace(**params[cli_args.layer_name])
    for k, v in vars(cli_args).items():
        if v is not None and k in params:
            setattr(args, k, v)
    old_files = [path for path in Path(args.input_folder).rglob('*' + args.ground_truth_file)]
    old_folders = [file.parent for file in old_files]
    evaluate(args, folder_list = old_folders)

