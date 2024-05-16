import argparse
from pathlib import Path
import pandas as pd
import yaml
from deckard.layers.afr import fit_aft, clean_data_for_aft, calculate_raw_failures


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--target', type=str, required=True, help='Target column name')
parser.add_argument('--duration_col', type=str, required=True, help='Duration column name')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

suffix = Path(args.data).suffix
if suffix == '.csv':
    df = pd.read_csv(args.data)
else:
    raise NotImplementedError(f'Unknown file format: {suffix}')

suffix = Path(args.config_file).suffix
assert Path(args.config_file).exists(), f"Model file not found: {args.config_file}"

if suffix == '.yaml':
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
else:
    raise NotImplementedError(f'Unknown file format: {suffix}')
assert args.model in config, f"Model not found in the model file: {args.model}"
# Config
dummies  = config.pop("dummies", {"atk_gen" : "Atk:", "def_gen" : "Def:", "id" : "Data:"})
covariate_list = config.get("covariates", [])
model_config = config[args.model].get('model', {})
# Will generate a subset used for fitting
data = df.copy()
# Calculate raw failures from accuracy
data = calculate_raw_failures(args, df, config)
from copy import deepcopy
ben_config = deepcopy(config)
ben_config['covariates'].remove('adv_failures')
ben_config['covariates'].append('ben_failures')
ben_config['covariates'].remove('accuracy')
ben_covariate_list = ben_config.get("covariates", [])
ben_data = deepcopy(data)
ben_failures = (1 - ben_data['accuracy']) * ben_data['attack.attack_size']
ben_data = ben_data.assign(ben_failures=ben_failures)
# Clean data by removing columns that are not in the covariate list and creating dummies for the categorical variables
ben_data = clean_data_for_aft(ben_data, ben_covariate_list, target='ben_failures', dummy_dict=dummies)
data = clean_data_for_aft(data, covariate_list, target=args.target, dummy_dict=dummies)

# Fit the model
model = fit_aft(data, event_col=args.target, duration_col=args.duration_col, **model_config, mtype=args.model)
# Predict the adversarial survival time
adv_survival_time = model.predict_expectation(data)
df = df.assign(adv_survival_time=adv_survival_time)
c_adv = df['train_time'] / df['adv_survival_time']
data['atk_value'] = 0
data[args.duration_col] = data['predict_time']
ben_model = fit_aft(ben_data, event_col='ben_failures', duration_col=args.duration_col, **model_config, mtype=args.model)
ben_survival_time = ben_model.predict_expectation(data)
df = df.assign(ben_survival_time=ben_survival_time)
c_ben = df['train_time'] / df['ben_survival_time']
df = df.assign(c_ben=c_ben, c_adv=c_adv)
suffix = Path(args.output).suffix
if suffix == '.csv':
    df.to_csv(args.output, index=False)
else:
    raise NotImplementedError(f'Unknown file format: {suffix}')
print(f"Output file: {args.output}")
