import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection


def load_data(path, delimiter):
    dataset = np.loadtxt(path, delimiter=delimiter)
    X = dataset[:, 1:3]
    y = dataset[:, 3]
    draw_raw_scatter_diagram(dataset, X, y)
    # extend the variable matrix to [x, 1]
    m, n = np.shape(X)
    X_extend = np.c_[X, np.ones(m)]

    return dataset, X_extend, y


def draw_raw_scatter_diagram(dataset, X, y):
    f = plt.figure(1)
    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='red', s=100, label='bad')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', s=100, label='good')
    plt.legend(loc='upper right')
    plt.show()


def sub_likelihood(X, y, beta_t):
    """
    Calculate the sub likelihood
    :param X: sample variables
    :param y: sample label
    :param beta_t: the transposed matrix of parameter vector
    :return: the sub log likelihood
    """
    return -y * np.dot(beta_t, X) + np.math.log(1 + np.math.exp(np.dot(beta_t, X)))


def likelihood(X, y, beta_t):
    """
    Calculate the total likelihood
    :param X: sample variables
    :param y: sample label
    :param beta_t: the transposed matrix of parameter vector
    :return: the total log likelihood
    """
    sum = 0
    m, n = np.shape(X)

    for i in range(m):
        sum += sub_likelihood(X[i], y[i], beta_t)

    return sum


def gradient_descent(X, y):
    """
    Implementation of functional gradient descent algorithms
    :param X: sample variables
    :param y: sample label
    :return: the best parameter estimate
    """
    # step length of iterator
    h = 0.1
    # give the iterative times limit
    max_times = 500
    m, n = np.shape(X)
    print(m, n)
    # for show convergence curve of parameter
    b = np.zeros((n, max_times))
    # parameter and initial
    beta = np.zeros(n)
    delta_beta = np.ones(n) * h
    lh = 0

    for i in range(max_times):
        beta_temp = beta.copy()
        for j in range(n):
            # for partial derivative
            beta[j] += delta_beta[j]
            likelihood_tmp = likelihood(X, y, beta)
            delta_beta[j] = -h * (likelihood_tmp - lh) / (delta_beta[j])
            b[j, i] = beta[j]
            beta[j] = beta_temp[j]
        beta += delta_beta
        lh = likelihood(X, y, beta)

    t = np.arange(max_times)
    f = plt.figure(3)

    p1 = plt.subplot(3, 1, 1)
    p1.plot(t, b[0])
    plt.ylabel('w1')

    p2 = plt.subplot(3, 1, 2)
    p2.plot(t, b[1])
    plt.ylabel('w2')

    p3 = plt.subplot(3, 1, 3)
    p3.plot(t, b[2])
    plt.ylabel('b')

    plt.show()

    return beta


def gradient_descent_stochastic(X, y):
    """
    Implementation of stochastic gradient descent algorithms
    :param X: sample variables
    :param y: sample label
    :return: the best parameter estimate
    """
    # step length of iterator
    h = 0.5
    # give the iterative times limit
    m, n = np.shape(X)
    # for show convergence curve of parameter
    b = np.zeros((n, m))
    # parameter and initial
    beta = np.zeros(n)
    delta_beta = np.ones(n) * h
    lh = 0

    for i in range(m):
        beta_temp = beta.copy()
        for j in range(n):
            # change step length of iterator
            h = 0.5 * 1 / (1 + i + j)
            # for partial derivative
            beta[j] += delta_beta[j]
            likelihood_tmp = sub_likelihood(X[i], y[i], beta)
            delta_beta[j] = -h * (likelihood_tmp - lh) / (delta_beta[j])
            b[j, i] = beta[j]
            beta[j] = beta_temp[j]
        beta += delta_beta
        lh = sub_likelihood(X[i], y[i], beta)

    t = np.arange(m)
    f = plt.figure(4)

    p1 = plt.subplot(3, 1, 1)
    p1.plot(t, b[0])
    plt.ylabel('w1')

    p2 = plt.subplot(3, 1, 2)
    p2.plot(t, b[1])
    plt.ylabel('w2')

    p3 = plt.subplot(3, 1, 3)
    p3.plot(t, b[2])
    plt.ylabel('b')

    plt.show()

    return beta


def sigmoid(x, beta):
    """
    Return the sigmoid function value
    :param x: the predict variable
    :param beta: the parameter
    :return: the sigmoid function value
    """
    return 1.0 / (1 + np.math.exp(- np.dot(beta, x.T)))


def predict(X, beta):
    m, n = np.shape(X)
    y = np.zeros(m)
    for i in range(m):
        if sigmoid(X[i], beta) > 0.5:
            y[i] = 1

    return y


def print_confusion_matrix(y_pred, y):
    # calculation of confusion_matrix and prediction accuracy
    cfmat = np.zeros((2, 2))
    for i in range(len(y)):
        if y_pred[i] == y[i] == 0:
            cfmat[0, 0] += 1
        elif y_pred[i] == y[i] == 1:
            cfmat[1, 1] += 1
        elif y_pred[i] == 0:
            cfmat[1, 0] += 1
        elif y_pred[i] == 1:
            cfmat[0, 1] += 1

    print(cfmat)


if __name__ == "__main__":
    dataset, X_extend, y = load_data('../data/watermelon_3a.csv', ',')

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_extend, y, test_size=0.5, random_state=0)

    beta1 = gradient_descent(X_train, y_train)
    y_pred1 = predict(X_test, beta1)
    print_confusion_matrix(y_pred1, y_test)

    beta2 = gradient_descent_stochastic(X_train, y_train)
    y_pred2 = predict(X_test, beta2)
    print_confusion_matrix(y_pred2, y_test)
