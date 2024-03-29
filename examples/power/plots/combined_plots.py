import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from deckard.layers.plots import set_matplotlib_vars

set_matplotlib_vars()


sns.set_theme(style="whitegrid", font_scale=1.2, font="times new roman")

normal_dir = "data"
datasets = ["mnist", "cifar", "cifar100"]
extra_data_dir = "bit_depth"

#
big_df = pd.DataFrame()
for data in datasets:
    df = pd.read_csv(
        Path(normal_dir) / data / "power.csv",
        index_col=0,
        low_memory=False,
    )
    df["dataset"] = data
    big_df = pd.concat([big_df, df], axis=0)
    if Path(normal_dir, extra_data_dir, data, "power.csv").exists():
        extra_df = pd.read_csv(
            Path(normal_dir, extra_data_dir, data, "power.csv"),
            index_col=0,
            low_memory=False,
        )
        extra_df["dataset"] = data
        big_df = pd.concat([big_df, extra_df], axis=0)


# Device Metadata
memory_bandwith = {
    "nvidia-tesla-p100": 732,
    "nvidia-tesla-v100": 900,
    "nvidia-l4": 250,
}
dataset_resolution = {
    "mnist": 28,
    "cifar": 32,
    "cifar100": 32,
}
dataset_channels = {
    "mnist": 1,
    "cifar": 3,
    "cifar100": 3,
}
dataset_classes = {
    "mnist": 10,
    "cifar": 10,
    "cifar100": 100,
}
cost = {
    "nvidia-tesla-p100": 1.60,
    "nvidia-tesla-v100": 2.55,
    "nvidia-l4": 0.81,
}
epochs = "model.trainer.nb_epoch"
batch_size = "model.trainer.batch_size"
bit_depth = "model.art.preprocessor.bit_depth"
resolution = "n_pixels"

# Add Metadata
for device in big_df.device_id.unique():
    big_df.loc[big_df.device_id == device, "peak_memory_bandwidth"] = float(
        memory_bandwith[device],
    )
    big_df.loc[big_df.device_id == device, "cost"] = float(cost[device])
big_df["train_cost"] = big_df["train_time"] * big_df["cost"]
big_df["predict_cost"] = big_df["predict_time"] * big_df["cost"]
big_df["adv_fit_cost"] = big_df["adv_fit_time"] * big_df["cost"]

ben_train_samples = 48000
ben_pred_samples = 12000
adv_pred_samples = big_df['attack.attack_size'].values
train_cost_per_sample = big_df['train_cost'] / ben_train_samples
predict_cost_per_sample = big_df['predict_cost'] / ben_pred_samples
adv_fit_cost_per_sample = big_df['adv_fit_cost'] / adv_pred_samples
train_power_per_sample = big_df['train_power'] / ben_train_samples
predict_power_per_sample = big_df['predict_power'] / ben_pred_samples
adv_fit_power_per_sample = big_df['adv_fit_power'] / adv_pred_samples
big_df = big_df.reset_index(drop=True)
big_df['train_cost_per_sample'] = train_cost_per_sample.values
big_df['predict_cost_per_sample'] = predict_cost_per_sample.values
big_df['adv_fit_cost_per_sample'] = adv_fit_cost_per_sample.values
big_df['train_power_per_sample'] = train_power_per_sample.values
big_df['predict_power_per_sample'] = predict_power_per_sample.values
big_df['adv_fit_power_per_sample'] = adv_fit_power_per_sample.values
big_df['train_time_per_sample'] = big_df['train_time'] / ben_train_samples
big_df['predict_time_per_sample'] = big_df['predict_time'] / ben_pred_samples
big_df['adv_fit_time_per_sample'] = big_df['adv_fit_time'] / adv_pred_samples

big_df["Device"] = big_df["device_id"].str.replace("-", " ").str.title()
big_df = big_df.reset_index(drop=True)
Path("data/combined").mkdir(parents=True, exist_ok=True)
Path("combined").mkdir(parents=True, exist_ok=True)
big_df.to_csv("data/combined/combined.csv")
big_df = pd.read_csv("data/combined/combined.csv", index_col=0, low_memory=False)

# Capitalize all letters in the dataset
big_df['dataset'] = big_df['dataset'].str.upper()
# Split the device_id on the last word and only keep the last word
big_df['Device'] = big_df['device_id'].str.split('-').str[-1].str.upper()
# Accuracy Plot
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ben_acc = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="accuracy",
    hue="Device",
    ax=ax[0],
)
ben_acc.set_title("")
ben_acc.set_ylabel("Ben. Accuracy")
ben_acc.set_xlabel("")
ben_acc.tick_params(axis='x', labelsize=12, rotation=0)
ben_acc.set_yscale("linear")
ben_acc.legend().remove()
adv_acc = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="adv_accuracy",
    hue="Device",
    ax=ax[1],
)
adv_acc.set_title("")
adv_acc.set_ylabel("Adv. Accuracy")
adv_acc.set_yscale("linear")
adv_acc.set_xlabel("")
adv_acc.tick_params(axis='x', labelsize=12, rotation=0)
xticklabels = [item.get_text() for item in adv_acc.get_xticklabels()]
adv_acc.legend()
# for _, ax in enumerate(fig.axes):
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
fig.tight_layout()
fig.savefig("combined/acc.pdf")


sns.set_theme(style="whitegrid", font_scale=1.8, font="times new roman")


# Time Plot
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
train_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="train_time_per_sample",
    hue="Device",
    ax=ax[0],
)
train_time.set_title("")
train_time.set_ylabel("$t_{t}$ (seconds)")
train_time.set_xlabel("")
train_time.tick_params(axis='x', labelsize=18)
train_time.legend().remove()
predict_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="predict_time_per_sample",
    hue="Device",
    ax=ax[1],
)
predict_time.set_title("")
predict_time.set_ylabel("$t_{i}$ (seconds)")
predict_time.set_xlabel("")
predict_time.tick_params(axis='x', labelsize=18)
predict_time.legend().remove()
adv_fit_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="adv_fit_time_per_sample",
    hue="Device",
    ax=ax[2],
)
adv_fit_time.set_title("")
adv_fit_time.set_ylabel("$t_{a}$ (seconds)")
adv_fit_time.set_xlabel("")
adv_fit_time.tick_params(axis='x', labelsize=18)
adv_fit_time.legend()
fig.tight_layout()
fig.savefig("combined/time.pdf")

# Power Plot
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
train_power = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="train_power_per_sample",
    hue="Device",
    ax=ax[0],
)
train_power.set_title("")
train_power.set_ylabel("$P_{t}$ (Watts)")
train_power.set_xlabel("")
train_power.tick_params(axis='x', labelsize=18)
train_power.legend().remove()
predict_power = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="predict_power_per_sample",
    hue="Device",
    ax=ax[1],
)
predict_power.set_title("")
predict_power.set_ylabel("$P_{i}$ (Watts)")
predict_power.set_xlabel("")
predict_power.tick_params(axis='x', labelsize=18)
predict_power.legend().remove()
adv_fit_power = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="adv_fit_power_per_sample",
    hue="Device",
    ax=ax[2],
)
adv_fit_power.set_title("")
adv_fit_power.set_ylabel("$P_{a}$ (Watts)")
adv_fit_power.set_xlabel("")
adv_fit_power.tick_params(axis='x', labelsize=18)
adv_fit_power.legend()
fig.tight_layout()
fig.savefig("combined/power.pdf")

# Cost Plot
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
train_cost = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="train_cost_per_sample",
    hue="Device",
    ax=ax[0],
)
train_cost.set_title("")
train_cost.set_ylabel("$C_{t}$ (USD)")
train_cost.set_xlabel("")
train_cost.tick_params(axis='x', labelsize=18)
train_cost.legend().remove()
predict_cost = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="predict_cost_per_sample",
    hue="Device",
    ax=ax[1],
)
predict_cost.set_title("")
predict_cost.set_ylabel("$C_{i}$ (USD)")
predict_cost.set_xlabel("")
predict_cost.tick_params(axis='x', labelsize=18)
predict_cost.legend().remove()
adv_fit_cost = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="adv_fit_cost_per_sample",
    hue="Device",
    ax=ax[2],
)
adv_fit_cost.set_title("")
adv_fit_cost.set_ylabel("$C_{a}$ (USD)")
adv_fit_cost.set_xlabel("")
adv_fit_cost.tick_params(axis='x', labelsize=18)
adv_fit_cost.legend()
fig.tight_layout()
fig.savefig("combined/cost.pdf")
