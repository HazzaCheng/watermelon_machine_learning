import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def load_data(path, delimiter):
    dataset = np.loadtxt(path, delimiter=delimiter)
    X = dataset[:, 1:3]
    y = dataset[:, 3]
    draw_raw_scatter_diagram(dataset, X, y)

    return dataset, X, y


def draw_raw_scatter_diagram(dataset, X, y):
    f = plt.figure(1)
    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='red', s=100, label='bad')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', s=100, label='good')
    plt.legend(loc='upper right')
    plt.show()


def logistic_regression(X, y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
    # training
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    # testing
    y_pred = lr_model.predict(X_test)

    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    draw_decision_boundary(lr_model, X, y)


def draw_decision_boundary(lr_model, X, y):
    f = plt.figure(2)
    h = 0.001
    x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                         np.arange(x1_min, x1_max, h))

    # your model's prediction (classification) function
    z = lr_model.predict(np.c_[x0.ravel(), x1.ravel()])

    # Put the result into a color plot
    z = z.reshape(x0.shape)
    plt.contourf(x0, x1, z, cmap=pl.cm.Paired)

    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='red', s=100, label='bad')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', s=100, label='good')
    plt.show()


if __name__ == '__main__':
    dataset, X_tmp, y_tmp = load_data('../data/watermelon_3a.csv', ',')
    logistic_regression(X_tmp, y_tmp)
