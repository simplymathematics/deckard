import optuna
import logging
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from ..base.utils import flatten_dict, unflatten_dict



def compose_experiment(config_dir, config_name, overrides):
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)
        cfg = OmegaConf.to_container(cfg, resolve=False)
    return cfg

def train_model(config_dir, config_name, overrides, data_file, model_file):
    cfg = compose_experiment(config_dir, config_name, overrides)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    exp = instantiate(cfg)
    scores = exp(model_file=model_file, data_file=data_file)
    return scores

def create_new_dataset(attack_samples_file, data_file, ratio = .5):
    attack_suffix = Path(attack_samples_file).suffix
    if attack_suffix == ".csv":
        attack_samples = pd.read_csv(attack_samples_file)
    elif attack_suffix == ".pkl":
        attack_samples = pd.read_pickle(attack_samples_file)
    else:
        raise ValueError(f"Unknown attack samples file format {attack_suffix}")
    data_suffix = Path(data_file).suffix
    if data_suffix == ".csv":
        data = pd.read_csv(data_file)
    elif data_suffix == ".pkl":
        data = pd.read_pickle(data_file)
    else:
        raise ValueError(f"Unknown data file format {data_suffix}")
    attack_len = len(attack_samples)
    data_len = len(data)
    new_len = int(attack_len * ratio)
