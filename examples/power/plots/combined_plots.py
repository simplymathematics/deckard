import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

normal_dir = "data"
datasets = ["mnist", "cifar", "cifar100"]
extra_data_dir = "bit_depth"


big_df = pd.DataFrame()
for data in datasets:
    df = pd.read_csv(
        Path(normal_dir) / data / "power.csv",
        index_col=0,
        low_memory=False,
    )
    df["dataset"] = data
    print(f"Shape of {data} is {df.shape}")
    big_df = pd.concat([big_df, df], axis=0)
    if Path(normal_dir, extra_data_dir, data, "power.csv").exists():
        extra_df = pd.read_csv(
            Path(normal_dir, extra_data_dir, data, "power.csv"),
            index_col=0,
            low_memory=False,
        )
        extra_df["dataset"] = data
        print(f"Shape of {extra_data_dir}/{data} is {extra_df.shape}")
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


plt.hist(big_df["predict_time"])


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


for device in big_df.device_id.unique():
    big_df.loc[big_df.device_id == device, "peak_memory_bandwith"] = float(
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

big_df["peak_memory_bandwith"] = big_df["peak_memory_bandwith"].astype(float)
big_df.loc[:, "memory_per_batch"] = (
    big_df[batch_size] * big_df[resolution] * big_df[resolution] * big_df[bit_depth] / 8
).values
big_df["Device"] = big_df["device_id"].str.replace("-", " ").str.title()
big_df = big_df.reset_index(drop=True)
Path("data/combined").mkdir(parents=True, exist_ok=True)
big_df.to_csv("data/combined/combined.csv")
big_df = pd.read_csv("data/combined/combined.csv", index_col=0, low_memory=False)


# acc_melt = pd.melt(big_df, id_vars=['name', 'device_id', 'dataset'], value_vars=['accuracy', 'adv_accuracy'], var_name='accuracy_type', value_name='accuracy_melt')
# pow_melt = pd.melt(big_df, id_vars=['name'], value_vars=['predict_power', 'train_power', 'adv_fit_power', 'adv_predict_power'], var_name='power_type', value_name='power_melt')
# time_melt = pd.melt(big_df, id_vars=['name'], value_vars=['predict_time', 'train_time', 'adv_fit_time', 'adv_predict_time'], var_name='time_type', value_name='time_melt')


fig, ax = plt.subplots(1, 2, figsize=(8, 5))
ben_acc = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="accuracy",
    hue="Device",
    ax=ax[0],
    legend=False,
)
ben_acc.set_title("Average Accuracy on Benign Samples")
ben_acc.set_ylabel("Ben. Accuracy")
ben_acc.set_xlabel("Dataset")
adv_acc = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="adv_accuracy",
    hue="Device",
    ax=ax[1],
)
adv_acc.set_title("Average Accuracy on Adversarial Samples")
adv_acc.set_ylabel("Adv. Accuracy")
adv_acc.set_xlabel("Dataset")

Path("plots/combined").mkdir(parents=True, exist_ok=True)
fig.savefig("plots/combined/acc.pdf")


fig, ax = plt.subplots(1, 3, figsize=(16, 5))
train_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="train_time",
    hue="Device",
    ax=ax[0],
)
train_time.set_title("Average Training Time per Sample")
train_time.set_ylabel("$t_{t}$ (seconds)")
train_time.set_xlabel("Dataset")
predict_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="predict_time",
    hue="Device",
    ax=ax[1],
)
predict_time.set_title("Average Inference Time per Sample")
predict_time.set_ylabel("$t_{i}$ (seconds)")
predict_time.set_xlabel("Dataset")
adv_fit_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="adv_fit_time",
    hue="Device",
    ax=ax[2],
)
adv_fit_time.set_title("Average Attack Time per Sample")
adv_fit_time.set_ylabel("$t_{a}$ (seconds)")
adv_fit_time.set_xlabel("Dataset")
fig.savefig("plots/combined/time.pdf")


fig, ax = plt.subplots(1, 3, figsize=(18, 5))
train_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="train_power",
    hue="Device",
    ax=ax[0],
)
train_time.set_title("Average Training Power per Sample")
train_time.set_ylabel("$P_{t}$ (Watts)")
train_time.set_xlabel("Dataset")
predict_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="predict_power",
    hue="Device",
    ax=ax[1],
)
predict_time.set_title("Average Inference Power per Sample")
predict_time.set_ylabel("$P_{i}$ (Watts)")
predict_time.set_xlabel("Dataset")
adv_fit_time = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="adv_fit_power",
    hue="Device",
    ax=ax[2],
)
adv_fit_time.set_title("Average Attack Power per Sample")
adv_fit_time.set_ylabel("$P_{a}$ (Watts)")
adv_fit_time.set_xlabel("Dataset")
fig.savefig("plots/combined/power.pdf")


fig, ax = plt.subplots(1, 3, figsize=(18, 5))
train_cost = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="train_cost",
    hue="Device",
    ax=ax[0],
)
train_cost.set_title("Average Training Cost per Sample")
train_cost.set_ylabel("$C_{t}$ (USD)")
train_cost.set_xlabel("Dataset")
predict_cost = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="predict_cost",
    hue="Device",
    ax=ax[1],
    legend=False,
)
predict_cost.set_title("Average Inference Cost per Sample")
predict_cost.set_ylabel("$C_{i}$ (USD)")
predict_cost.set_xlabel("Dataset")
adv_fit_cost = sns.boxenplot(
    data=big_df,
    x="dataset",
    y="adv_fit_cost",
    hue="Device",
    ax=ax[2],
)
adv_fit_cost.set_title("Average Attack Cost per Sample")
adv_fit_cost.set_ylabel("$C_{a}$ (USD)")
adv_fit_cost.set_xlabel("Dataset")
fig.savefig("plots/combined/cost.pdf")
