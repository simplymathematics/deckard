# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# Modified for thesis by Charles Meyers
# License: BSD 3 clause

import logging
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons, make_regression, load_digits, fetch_kddcup99
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

import pandas as pd


binarizer = lambda y: 0 if y <=.5 else 1
binary_func = np.vectorize(binarizer)

def plot_dataset_model_comparison(names, classifiers, datasets, dataset_names, file="thesis/figures/dataset_model_comparison.pdf"):
    figure = plt.figure(figsize=(22, 7))
    i = 1
    # iterate over datasets
    print("Plotting dataset model comparison")
    for ds_cnt, ds in tqdm(enumerate(datasets), desc="Datasets", position=0, leave=True):
        # preprocess dataset, split into training and test part
        X, y = ds
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=100, train_size=100, random_state=42
        ) 

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data", fontsize=14)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        # Plot the testing points
        ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_ylabel(f"{dataset_names[ds_cnt]}", fontsize=14)
        i += 1

        # iterate over classifiers
        for name, clf in tqdm(zip(names, classifiers), desc="Classifiers", position=1, leave=False):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            preds = binary_func(preds)
            score = accuracy_score(y_test, preds)
            DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5, response_method="predict",
            )

            # Plot the training points
            ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
            )
            # Plot the testing points
            ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
            )
            # Specify the limits of each plot
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0: # Otherwise, this is handled outside of this loop
                ax.set_title(name, fontsize=14)
            ax.text(
            x_max - 0.2,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"), 
            size=15,
            horizontalalignment="right",
            ) # Display the accuracy score
            i += 1
    # Add title to the entire figure
    figure.suptitle("Classifiers Comparison", fontsize=18)
    # Add label to the y axis
    figure.supylabel("Dataset", fontsize=18)
    # Shift the y axis label to the right
    figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # Add label to the x axis
    figure.supxlabel("Model", fontsize=18)
    # # Create legend handles using the cm_bright colormap
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="b", markersize=10),
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markersize=10),
    ]
    labels = ["Class 0", "Class 1", "Uncertain"]
    plt.legend(handles, labels, loc="lower left", bbox_to_anchor=(1.05, 2.5), fontsize=15)
    # Layout the figure
    figure.tight_layout()
    # Save the figure
    if file is not None:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(file)

if "__main__" == __name__:
    
    ##############################################################################
    # Model list
    names = [
    "Linear Regression",
    "Logistic Regression",
    "KNN",
    "Linear SVM",
    "RBF SVM",
    "Neural Net",
    ]

    classifiers = [
        LinearRegression(),
        LogisticRegression(),
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=42),
        SVC(gamma=2, C=1, random_state=42),
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    ]

    
    ##############################################################################
    # Linearly Separable Classification
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1, scale=1.0, class_sep=10, n_samples=200,
    )
    linearly_separable = (X, y)
    print("Linearly Separable Classification dataset created.")

    # Linearly Separable Regression
    X,y = make_regression(n_features=2, n_informative=2, random_state=1, noise=1, n_samples=200)
    y = binary_func(y)
    regression = (X, y)
    print("Linearly Separable Regression dataset created.")
    
    # MNIST (PCA reduced to 2 dimensions)
    digits = load_digits()
    X = digits.data[(digits.target == 0) | (digits.target == 1)]
    y = digits.target[(digits.target == 0) | (digits.target == 1)]
    X, _, y, _ = train_test_split(X, y, test_size=160, train_size=200, random_state=42)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    mnist = (X_pca, y)
    print("MNIST dataset created.")

    # # kddcup99
    # kddcup99 = fetch_kddcup99(data_home="thesis/raw_data")
    # X = kddcup99.data
    # y = kddcup99.target
    # # Transform the target to binary
    # kdd_binarizer = lambda y: 0 if y == b"normal." else 1
    # kdd_bin_func = np.vectorize(kdd_binarizer)
    # y = kdd_bin_func(y)
    # # Transform the input into numeric values using pandas get_dummies
    # X = pd.get_dummies(pd.DataFrame(X)).values
    # X, _, y, _ = train_test_split(X, y, test_size=100, train_size=100, random_state=42)
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    # kddcup99 = (X_pca, y)
    # print("KDDCup99 dataset created.")

    # Dataset list
    datasets = [
        linearly_separable,
        regression, 
        make_moons(noise=0.3, random_state=0, n_samples=200,),
        make_circles(noise=0.05, factor=0.5, random_state=1, n_samples=200,),
        mnist,
        # kddcup99,
    ]

    # Dataset names list
    dataset_names = [
        "Linearly\nSeparable\nClassification",
        "Linearly\nSeparable\nRegression",
        "Clusters",
        "Circles",
        "MNIST",
        # "KDDCup99",
    ]
    
    plot_dataset_model_comparison(names, classifiers, datasets, dataset_names)
    