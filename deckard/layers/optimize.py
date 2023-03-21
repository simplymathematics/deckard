import logging
import os
import yaml
import json
from pathlib import Path
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from deckard.layers.runner import load_dvc_experiment
from deckard.layers.parse import parse

logger = logging.getLogger(__name__)
os.environ['HYDRA_FULL_ERROR'] = '1'


@hydra.main(
    version_base=None,
    config_path=str(Path(os.getcwd(), "conf").as_posix()),
    config_name="config.yaml",
)
def hydra_optimizer(cfg:DictConfig)->None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    if "stage" in cfg:
        stage = cfg['stage']
        del cfg['stage']
    else:
        raise ValueError("Stage not specified")
    if "direction" in cfg:
        direction = cfg['direction']
        if direction == "maximize":
            direction = True
        elif direction == "minimize":
            direction = False
        del cfg['direction']
    else:
        direction = True # True = maximize, False = minimize
    return optimize(cfg, stage=stage, scorer_maximize=direction)

def optimize(params:dict, stage:str=None, scorer=None, scorer_maximize:bool=False):
    # Stage selects a subset of the pipeline to run (i.e. number of layers to run inside a single container)
    assert stage is not None, "Stage not specified"
    if scorer_maximize is True:
        best_score = -1e10
    else:
        best_score = 1e10
    parsed_df = parse(params, stage=stage)
    for entry in parsed_df.iterrows():
        entry = entry[1].to_dict()
        exp = load_dvc_experiment(stage=stage, params=entry, mode='hydra')
        results = exp.run(save_model=False)
        with open(Path(results['scores'])) as f:
            score_dict = json.load(f)
        if scorer is not None:
            score = scorer(score_dict)
        else:
            score = list(score_dict.values())[0]
        if score <= best_score and scorer_maximize is False:
            best_score = score
        elif score >= best_score and scorer_maximize is True:
            best_score = score
    return best_score
    
    

    
    
if "__main__" == __name__:
    hydra_optimizer()