import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


def draw_iris_pairplot():
    sns.set(style='white', color_codes=True)
    iris = sns.load_dataset('iris')
    sns.pairplot(iris, hue='species', diag_kind='auto')
    plt.show()


def draw_transfusion_pairplot(path):
    sns.set(style='white', color_codes=True)
    transfusion = pd.read_csv(path, skiprows=1)
    transfusion.columns = ['Recency', 'Frequency', 'Monetary', 'Time', 'Donate']
    sns.pairplot(transfusion, vars=transfusion.columns[0:4], hue='Donate', diag_kind='hist')
    plt.show()


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


def logistic_regression_k_folds(X, y, k):
    # m = np.shape(X)[0]
    lr_model = LogisticRegression(solver='lbfgs')
    y_pred = cross_val_predict(lr_model, X, y, cv=k)

    print('Confusion Matrix\n', metrics.confusion_matrix(y, y_pred))
    print('Accuracy Score\n', metrics.accuracy_score(y, y_pred))
    print('Classification Report\n', metrics.classification_report(y, y_pred))


if __name__ == "__main__":
    iris_dataset, X_iris, y_iris = load_iris_dataset()
    transfusion_dataset, X_transfusion, y_transfusion = load_transfusion_dataset('../data/transfusion.data', ',')
    draw_iris_pairplot()
    draw_transfusion_pairplot('../data/transfusion.data')
    # 10 folds
    logistic_regression_k_folds(X_iris, y_iris, 10)
    logistic_regression_k_folds(X_transfusion, y_transfusion, 10)
