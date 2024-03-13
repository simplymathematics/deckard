import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


try:
    import plotext as plt

    plot = True
except ImportError:
    logger.info(
        "Plotext not installed. Please install plotext to use the plotting functions. Will skip the plotting for now.",
    )
    plot = False

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path


def undersample(df, target, n_samples=10000):
    """
    Undersamples the dataframe to balance the target column
    """
    y = df[target]
    columns = list(df.columns)
    X = df.drop(target, axis=1)
    n_classes = y.value_counts().shape[0]
    keys = y.value_counts().keys()
    values = [n_samples // n_classes] * len(keys)
    sampling_strategy = dict(zip(keys, values))
    rus = RandomUnderSampler(random_state=0, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    return df_resampled


def undersample_datasets(datasets: list, targets: list, n_samples: list):
    """
    Undersamples the datasets to balance the target columns
    """
    assert len(datasets) == len(
        targets,
    ), "The number of datasets and targets must be the same"
    for i in range(len(datasets)):
        path = Path(datasets[i])
        print(f"Undersampling {path.as_posix()}")
        df = pd.read_csv(datasets[i])
        df = undersample(df, targets[i], n_samples[i])
        new_name = path.stem + f"_undersampled_{n_samples[i]}.csv"
        new_name = path.parent / new_name
        print(f"Renaming to {new_name}")
        Path(new_name).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(new_name, index=False)
        print(f"Saved to {new_name}")


if __name__ == "__main__":

    Path("raw_data").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        "https://gist.githubusercontent.com/simplymathematics/8c6c04bd151950d5ea9e62825db97fdd/raw/34e546e4813f154d11d4f13869b9e3481fc3e829/kdd_nsl.csv",
    )
    del df["difficulty_level"]
    X = df.drop("label", axis=1)
    y = df["label"]
    y = pd.DataFrame(y, columns=["label"])
    df = pd.concat([X, y], axis=1)
    df.to_csv("raw_data/kdd_nsl.csv", index=False)
    # Find the number of entries for each label
    counts = pd.DataFrame(df["label"]).value_counts().values
    labels = range(len(counts))
    # Plot the counts
    if plot is True:
        plt.simple_bar(labels, counts, title="KDD NSL Label Counts", width=50)
        plt.show()
    else:
        logger.info("Label counts for KDD NSL: {}".format(counts))
    df = pd.read_csv(
        "https://gist.githubusercontent.com/simplymathematics/8c6c04bd151950d5ea9e62825db97fdd/raw/34e546e4813f154d11d4f13869b9e3481fc3e829/truthseeker.csv",
    )
    X = df["tweet"]
    label = "BotScoreBinary"
    y = df[label]
    df = pd.concat([X, y], axis=1)
    df.to_csv("raw_data/truthseeker.csv", index=False)
    # Find the number of entries for each label
    counts = pd.DataFrame(df[label]).value_counts().values
    labels = range(len(counts))
    if plot is True:
        # Plot the counts
        plt.simple_bar(labels, counts, title="Truthseeker Label Counts", width=50)
        plt.show()
    else:
        logger.info("Label counts for Truthseeker: {}".format(counts))
    df = pd.read_csv(
        "https://gist.githubusercontent.com/simplymathematics/8c6c04bd151950d5ea9e62825db97fdd/raw/c91944733b8f2b9a6ac0b8c8fab01ddcdf0898eb/sms-spam.csv",
    )
    X = df["message"]
    y = df["label"]
    y = y.str.replace("ham", "0").replace("spam", "1")
    df = pd.concat([X, y], axis=1)
    df.to_csv("raw_data/sms-spam.csv", index=False)
    # Find the number of entries for each label
    counts = pd.DataFrame(df["label"]).value_counts().values
    labels = range(len(counts))
    # Plot the counts
    if plot is True:
        plt.simple_bar(labels, counts, title="SMS Spam Label Counts", width=50)
        plt.show()
    else:
        logger.info("Label counts for SMS Spam: {}".format(counts))
    df = pd.read_csv(
        "https://gist.githubusercontent.com/simplymathematics/8c6c04bd151950d5ea9e62825db97fdd/raw/712b528dcd212d5a6d1767332f50161fc1cfe55c/ddos.csv",
    )
    # Find the number of entries for each label
    X = df.drop("Label", axis=1)
    y = df["Label"]
    y = y.str.replace("Benign", "0").replace("ddos", "1")
    df = pd.concat([X, y], axis=1)
    df.to_csv("raw_data/ddos.csv", index=False)
    counts = pd.DataFrame(y).value_counts().values
    labels = range(len(counts))
    # Plot the counts
    if plot is True:
        plt.simple_bar(labels, counts, title="DDoS Label Counts", width=50)
        plt.show()
    else:
        logger.info("Label counts for DDoS: {}".format(counts))

    datasets = [
        "raw_data/kdd_nsl.csv",
        "raw_data/truthseeker.csv",
        "raw_data/sms-spam.csv",
        "raw_data/ddos.csv",
    ]
    targets = [
        "label",  # kdd_nsl
        "BotScoreBinary",  # truthseeker
        "label",  # sms_spam
        "Label",  # ddos
    ]
    n_samples = [
        5000,  # kdd_nsl
        8000,  # truthseeker
        1450,  # sms_spam
        10000,  # ddos
    ]

    paths = undersample_datasets(datasets, targets, n_samples)
