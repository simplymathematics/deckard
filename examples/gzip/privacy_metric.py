from gzip_classifier import GzipSVC
import art
import numpy as np
from gzip_classifier import prepare_data


# Create a GzipKNN classifier
def calculate_privacy_risk(X_train, X_test, y_train, y_test, metric):
    clf = GzipSVC(metric=metric)
    clf.fit(X_train, y_train)
    est = art.estimators.classification.SklearnClassifier(model=clf, preprocessing=None)
    privacy_risk = art.metrics.SHAPr(
        target_estimator=est,
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    accuracies = clf.score(X_test, y_test)
    return privacy_risk, accuracies


def line_search_metrics(X_train, X_test, y_train, y_test, metrics):
    metric_dict = {}
    acc_list = []
    shapr_list = []
    for metric in metrics:
        scores, privacy_risks = calculate_privacy_risk(
            X_train,
            X_test,
            y_train,
            y_test,
            metric,
        )
        metric_dict[metric] = {}
        nb_classes = len(np.unique(y_test))
        shaprs = []
        accuracies = []
        for i in range(nb_classes):
            idxs = np.where(y_test == i)
            privacy_risk = np.mean(privacy_risks[idxs])
            privacy_risk = 0.01 if privacy_risk == 0 else privacy_risk
            shaprs.append(privacy_risk)
            accuracy = round(np.mean(scores[idxs]), 2)
            accuracies.append(accuracy)
            print(
                f"{metric.capitalize()} SHAPr for class {i}: {privacy_risk}; Accuracy: {accuracy}",
            )
        assert len(shaprs) == nb_classes
        assert len(accuracies) == nb_classes
        acc_list.append(accuracies)
        shapr_list.append(shaprs)
        print(f"{metric.capitalize()} SHAPr: {shaprs}; Accuracy: {accuracies}")
    import plotext as plt

    label_prefixes = ["Accuracy", "SHAPr"]
    labels = [
        f"{label_prefixes[i]}_{j}"
        for i in range(nb_classes)
        for j in range(len(label_prefixes))
    ]
    values = []
    # Turn acc_list into separate lists for each class
    for i in range(nb_classes):
        sub_values = [acc_list[j][i] for j in range(len(acc_list))]
        values.append(sub_values)
    for i in range(nb_classes):
        sub_values = [shapr_list[j][i] for j in range(len(shapr_list))]
        values.append(sub_values)
    plt.simple_multiple_bar(
        metrics,
        values,
        title="Accuracy and SHAPr for different metrics",
        labels=labels,
    )
    plt.show()
    return metric_dict


if __name__ == "__main__":
    metrics = [
        "gzip",
        "lzma",
        "bz2",
        "zstd",
        "pkl",
        "levenshtein",
        "ratio",
        "hamming",
        "jaro",
        "jaro_winkler",
        "seqratio",
    ]
    datasets = ["kdd_nsl", "truthseeker", "sms-spam", "ddos"]

    for dataset in datasets:
        X_train, X_test, y_train, y_test = prepare_data(dataset=dataset)
        line_search_metrics(X_train, X_test, y_train, y_test, metrics)


# Decorator to make to turn a fit function into the batch'd version
