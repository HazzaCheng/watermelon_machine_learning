import numpy as np
from matplotlib import pyplot as plt


def load_data(path, delimiter):
    dataset = np.loadtxt(path, delimiter=delimiter)
    X = dataset[:, 1:3]
    y = dataset[:, 3]
    # delete outliner: line 10 and line 14
    X = np.delete(X, [9, 14], 0)
    y = np.delete(y, [9, 14], 0)
    draw_raw_scatter_diagram(X, y)

    return X, y


def draw_raw_scatter_diagram(X, y):
    f1 = plt.figure(1)
    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='red', s=100, label='bad')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', s=100, label='good')
    plt.legend(loc='upper right')
    plt.show()


def get_column_mean(X, y):
    """
    Get mean vector of each class.
    :param X:
    :param y:
    :return:
    """
    mu = []
    for i in range(2):
        # y has two class
        # get column mean
        mu.append(np.mean(X[y == i], axis=0))

    return mu


def get_within_class_scatter_matrix(X, y, mu):
    """
    Compute the within-class scatter matrix.
    :param X:
    :param y:
    :param mu:
    :return:
    """
    m, n = np.shape(X)
    Sw = np.zeros((n, n))
    # get mu_0 and mu_1
    mu_0 = mu[0].reshape(n, 1)
    mu_1 = mu[1].reshape(n, 1)
    mu_tmp = 0

    for i in range(m):
        # change row vector to column vector
        x_tmp = X[i].reshape(n, 1)
        if y[i] == 0:
            # get mu_0
            mu_tmp = mu_0
        if y[i] == 1:
            # get mu_1
            mu_tmp = mu_1
        Sw += np.dot((x_tmp - mu_tmp), (x_tmp - mu_tmp).T)

    return Sw


def get_W(Sw, mu_0, mu_1):
    """
    Compute the W.
    :param Sw
    :param mu_0:
    :param mu_1:
    :return:
    """
    m, n = np.shape(Sw)
    Sw = np.mat(Sw)
    U, sigma, V = np.linalg.svd(Sw)
    Sw_inv = U.T.dot(np.linalg.inv(np.diag(sigma))).dot(V.T)
    w = np.dot(Sw_inv, (mu_0 - mu_1).reshape((n, 1)))

    return w


def linear_discriminant_analysis(X, y):
    mu = get_column_mean(X, y)
    Sw = get_within_class_scatter_matrix(X, y, mu)
    w = get_W(Sw, mu[0], mu[1])
    print(w)
    draw_projective_point_on_lda_line(X, y, w)


def get_projective_point(point, line):
    a = point[0]
    b = point[1]
    k = line[0]
    t = line[1]

    if k == 0:
        return [a, t]
    elif k == np.inf:
        return [0, b]
    x = (a + k * b - k * t) / (k * k + 1)
    y = k * x + t
    return [x, y]


def draw_projective_point_on_lda_line(X, y, w):
    """
    Draw projective point on the line.
    :param X:
    :param y:
    :param w:
    :return:
    """
    f2 = plt.figure(2)
    plt.xlim(-0.2, 1)
    plt.ylim(-0.5, 0.7)

    p0_x = -X[:, 0].max()
    p0_y = (w[1, 0] / w[0, 0]) * p0_x
    p1_x = X[:, 0].max()
    p1_y = (w[1, 0] / w[0, 0]) * p1_x

    plt.title('watermelon_3a - LDA')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=10, label='bad')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=10, label='good')
    plt.legend(loc='upper right')

    plt.plot([p0_x, p1_x], [p0_y, p1_y])

    m, n = np.shape(X)
    for i in range(m):
        x_p = get_projective_point([X[i, 0], X[i, 1]], [w[1, 0] / w[0, 0], 0])
        if y[i] == 0:
            plt.plot(x_p[0], x_p[1], 'ko', markersize=5)
        if y[i] == 1:
            plt.plot(x_p[0], x_p[1], 'go', markersize=5)
        plt.plot([x_p[0], X[i, 0]], [x_p[1], X[i, 1]], 'c--', linewidth=0.3)

    plt.show()


if __name__ == '__main__':
    X, y = load_data('../data/watermelon_3a.csv', ',')
    linear_discriminant_analysis(X, y)
