import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from deckard.layers.plots import set_matplotlib_vars

set_matplotlib_vars()


sns.set_theme(style="whitegrid", font_scale=1.8, font="times new roman")

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


# if "l4" in big_df.device_id.str.lower().unique():
ben_train_samples = pd.Series(big_df["train_time"] / big_df["train_time_per_sample"])
ben_pred_samples = pd.Series(big_df["predict_time"] / big_df["predict_time_per_sample"])
adv_pred_samples = pd.Series(
    big_df["adv_predict_time"] / big_df["adv_predict_time_per_sample"],
)
big_df = big_df.assign(ben_pred_samples=ben_pred_samples.values)
big_df = big_df.assign(adv_pred_samples=adv_pred_samples.values)
big_df = big_df.assign(ben_train_samples=ben_train_samples.values)
big_df["train_time"] = big_df["train_time"] / big_df["ben_train_samples"]
big_df["predict_time"] = big_df["predict_time"] / (big_df["ben_pred_samples"] * 0.25)
big_df["adv_fit_time"] = big_df["adv_fit_time"] / big_df["adv_pred_samples"]
big_df["train_power"] = big_df["train_power"] / big_df["ben_train_samples"]
big_df["predict_power"] = big_df["predict_power"] / big_df["ben_pred_samples"]
big_df["adv_fit_power"] = big_df["adv_fit_power"] / big_df["adv_pred_samples"]


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
for dataset in big_df.dataset.unique():
    big_df.loc[big_df.dataset == dataset, "n_pixels"] = int(
        dataset_resolution[dataset] ** 2,
    )
    big_df.loc[big_df.dataset == dataset, "n_channels"] = int(dataset_channels[dataset])
    big_df.loc[big_df.dataset == dataset, "n_classes"] = int(dataset_classes[dataset])

big_df.loc[:, "memory_per_batch"] = (
    big_df[batch_size] * big_df[resolution] * big_df[resolution] * big_df[bit_depth] / 8
).values
big_df["Device"] = big_df["device_id"].str.replace("-", " ").str.title()
big_df = big_df.reset_index(drop=True)
Path("data/combined").mkdir(parents=True, exist_ok=True)
Path("plots/combined").mkdir(parents=True, exist_ok=True)
big_df.to_csv("data/combined/combined.csv")
big_df = pd.read_csv("data/combined/combined.csv", index_col=0, low_memory=False)

# Accuracy Plot
fig, ax = plt.subplots(1, 2, figsize=(8, 5))
ben_acc = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="accuracy",
    hue="Device",
    ax=ax[0],
)
ben_acc.set_title("")
ben_acc.set_ylabel("Ben. Accuracy")
ben_acc.set_xlabel("Dataset")
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
adv_acc.set_xlabel("Dataset")
adv_acc.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
for _, ax in enumerate(fig.axes):
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
fig.tight_layout()
fig.savefig("plots/combined/acc.pdf")

# Time Plot
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
train_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="train_time",
    hue="Device",
    ax=ax[0],
)
train_time.set_title("")
train_time.set_ylabel("$t_{t}$ (seconds)")
train_time.set_xlabel("Dataset")
train_time.legend().remove()
predict_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="predict_time",
    hue="Device",
    ax=ax[1],
)
predict_time.set_title("")
predict_time.set_ylabel("$t_{i}$ (seconds)")
predict_time.set_xlabel("Dataset")
predict_time.legend().remove()
adv_fit_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="adv_fit_time",
    hue="Device",
    ax=ax[2],
)
adv_fit_time.set_title("")
adv_fit_time.set_ylabel("$t_{a}$ (seconds)")
adv_fit_time.set_xlabel("Dataset")
adv_fit_time.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
fig.tight_layout()
fig.savefig("plots/combined/time.pdf")

# Power Plot
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
train_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="train_power",
    hue="Device",
    ax=ax[0],
)
train_time.set_title("")
train_time.set_ylabel("$P_{t}$ (Watts)")
train_time.set_xlabel("Dataset")
train_time.legend().remove()
predict_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="predict_power",
    hue="Device",
    ax=ax[1],
)
predict_time.set_title("")
predict_time.set_ylabel("$P_{i}$ (Watts)")
predict_time.set_xlabel("Dataset")
predict_time.legend().remove()
adv_fit_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="adv_fit_power",
    hue="Device",
    ax=ax[2],
)
adv_fit_time.set_title("")
adv_fit_time.set_ylabel("$P_{a}$ (Watts)")
adv_fit_time.set_xlabel("Dataset")
adv_fit_time.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
fig.tight_layout()
fig.savefig("plots/combined/power.pdf")

# Cost Plot
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
train_cost = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="train_cost",
    hue="Device",
    ax=ax[0],
)
train_cost.set_title("")
train_cost.set_ylabel("$C_{t}$ (USD)")
train_cost.set_xlabel("Dataset")
train_cost.legend().remove()
predict_cost = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="predict_cost",
    hue="Device",
    ax=ax[1],
)
predict_cost.set_title("")
predict_cost.set_ylabel("$C_{i}$ (USD)")
predict_cost.set_xlabel("Dataset")
predict_cost.legend().remove()
adv_fit_cost = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="adv_fit_cost",
    hue="Device",
    ax=ax[2],
)
adv_fit_cost.set_title("")
adv_fit_cost.set_ylabel("$C_{a}$ (USD)")
adv_fit_cost.set_xlabel("Dataset")
adv_fit_cost.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
fig.tight_layout()
fig.savefig("plots/combined/cost.pdf")
