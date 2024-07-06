import argparse
from pathlib import Path
import pandas as pd
import yaml
from deckard.layers.afr import fit_aft, clean_data_for_aft, calculate_raw_failures


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("--target", type=str, required=True, help="Target column name")
parser.add_argument(
    "--duration_col",
    type=str,
    required=True,
    help="Duration column name",
)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--training_samples", type=int, default=48000)
parser.add_argument("--attack_samples", type=int, default=100)
args = parser.parse_args()

suffix = Path(args.data).suffix
if suffix == ".csv":
    df = pd.read_csv(args.data)
else:
    raise NotImplementedError(f"Unknown file format: {suffix}")

suffix = Path(args.config_file).suffix
assert Path(args.config_file).exists(), f"Model file not found: {args.config_file}"

if suffix == ".yaml":
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
else:
    raise NotImplementedError(f"Unknown file format: {suffix}")
assert args.model in config, f"Model not found in the model file: {args.model}"
# Config
dummies = config.pop("dummies", {"atk_gen": "Atk:", "def_gen": "Def:", "id": "Data:"})
covariate_list = config.get("covariates", [])
model_config = config[args.model].get("model", {})
# Will generate a subset used for fitting
data = df.copy()
# Calculate raw failures from accuracy
data = calculate_raw_failures(args, df, config)
# Clean data by removing columns that are not in the covariate list and creating dummies for the categorical variables
data = clean_data_for_aft(data, covariate_list, target=args.target, dummy_dict=dummies)
# Fit the model
model = fit_aft(
    data,
    event_col=args.target,
    duration_col=args.duration_col,
    **model_config,
    mtype=args.model,
)
# Predict the adversarial survival time
adv_survival_time = model.predict_expectation(data)
df = df.assign(adv_survival_time=adv_survival_time)
train_rate = df["train_time_per_sample"]
normalized_survival_time = df["adv_survival_time"] / args.attack_samples
c_adv = train_rate / normalized_survival_time
df = df.assign(c_adv=c_adv)
assert "c_adv" in df.columns, "c_adv column not found in the dataframe"
suffix = Path(args.output).suffix
if suffix == ".csv":
    df.to_csv(args.output, index=False)
else:
    raise NotImplementedError(f"Unknown file format: {suffix}")
print(f"Output file: {args.output}")
