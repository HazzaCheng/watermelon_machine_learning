import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut


def load_iris_dataset():
    iris = sns.load_dataset("iris")
    # choose 100 records
    X = iris.values[:, 0:4]
    y = iris.values[:, 4]
    return iris, X, y


def load_transfusion_dataset(path, delim):
    # ignore the first line
    dataset = np.loadtxt(path, delimiter=delim, skiprows=1)
    X = dataset[50:100, 0:4]
    y = dataset[50:100, 4]
    return dataset, X, y


def logistic_regression_leave_one_out(X, y):
    lr_model = LogisticRegression(solver='lbfgs')
    loo = LeaveOneOut()
    accuracy = 0

    for train, test in loo.split(X):
        # train
        lr_model.fit(X[train], y[train])
        y_pred = lr_model.predict(X[test])
        if y_pred == y[test]:
            accuracy += 1
    print('Accuracy:', accuracy / np.shape(X)[0])


if __name__ == '__main__':
    iris_dataset, X_iris, y_iris = load_iris_dataset()
    transfusion_dataset, X_transfusion, y_transfusion = load_transfusion_dataset('../data/transfusion.data', ',')
    logistic_regression_leave_one_out(X_iris, y_iris)
    logistic_regression_leave_one_out(X_transfusion, y_transfusion)
