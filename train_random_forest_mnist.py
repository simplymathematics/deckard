# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.datasets    import load_digits

def main():
    # load data
    digits = load_digits()
    # split data into features and labels
    data = digits.data
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # train model
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    # make predictions
    y_pred = model.predict(X_test)
    # evaluate model
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    assert isinstance(model, RandomForestClassifier)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(accuracy_score(y_test, y_pred), float)
    assert isinstance(classification_report(y_test, y_pred), str)


if __name__ == '__main__':
    main()

    assert isinstance(main(), object)
    print('Test passed!')