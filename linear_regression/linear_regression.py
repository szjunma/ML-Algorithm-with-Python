import numpy as np
import matplotlib.pyplot as plt
import os, sys

def cost(X, y, theta):
    h = np.dot(X, theta)
    cos = np.sum((h - y) * ( h - y))/(2 * len(y))
    return cos

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    costs = []
    for _ in range(num_iters):
        h = np.dot(X,theta)
        theta -= alpha * (np.sum((h - y) * X, axis = 0)).reshape(X.shape[1], 1)/m
        costs.append(cost(X, y, theta))
    return theta, costs

def linear_regression(X, y, alpha = 0.01,num_iters = 100):
    X = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
    theta = np.zeros((X.shape[1], 1), dtype = np.float64)
    theta, costs = gradient_descent(X, y, theta, alpha, num_iters)
    predicted = np.dot(X, theta)
    return predicted, theta, costs

if __name__ == '__main__':
    images_dir = os.path.join(sys.path[0], 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    X = np.array(range(25))
    y = (X ** 1.3 + np.random.normal(10, 10, X.shape[0]))
    X, y = X.reshape((-1, 1)), y.reshape((-1, 1))
    # data = np.loadtxt(os.path.join(sys.path[0], 'linear_regression_data.csv'), delimiter = ',', dtype = np.float64)
    # X, y = data[:, :-1], data[:, -1].reshape((-1, 1))

    predicted, theta, costs = linear_regression(X, y)

    plt.plot(X, y, 'rx', 10)
    plt.title('Dataset')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.savefig(os.path.join(images_dir, 'data.png'))
    plt.clf()

    plt.plot(costs, 'b')
    plt.title('Cost vs Number of Interations')
    plt.ylabel('Cost')
    plt.xlabel('No. of Interations')
    plt.savefig(os.path.join(images_dir, 'cost.png'))
    plt.clf()

    plt.plot(X, predicted, 'b')
    plt.plot(X, y, 'rx', 10)
    for i, x in enumerate(X):
        plt.vlines(x, min(predicted[i], y[i]), max(predicted[i], y[i]))
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title('Error')
    plt.legend(('linear fit', 'data'))
    plt.savefig(os.path.join(images_dir, 'Error.png'))
    plt.clf()
