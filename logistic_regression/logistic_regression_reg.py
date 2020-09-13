import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from logistic_regression import *

def cost_reg(theta, X, y, lam = 0):
    h = sigmoid(np.dot(X, theta))
    theta1 = theta.copy()
    theta1[0] = 0
    cos = -(np.sum(y * np.log(h)) + np.sum((1 - y) * np.log(1 - h)))/len(y) + lam * np.sum(theta1 * theta1)/len(y)
    return cos

def gradient_descent_reg(X, y, theta, alpha, lam = 0, num_iters = 100):
    costs = []

    for _ in range(num_iters):
        h = sigmoid(np.dot(X, theta))
        theta1 = theta.copy()
        theta1[0] = 0
        theta -= alpha * (np.dot(X.T, (h - y)) + 2 * lam * theta1)/len(y)
        costs.append(cost_reg(theta, X, y))
    return theta, costs

def logistic_regression_reg(X, y, power = 2, alpha = 0.01, lam = 0, num_iters = 100):
    X = expand_feature(X[:, 0], X[:, 1], power = power)
    theta = np.zeros((X.shape[1], y.shape[1]), dtype = np.float64)
    theta, costs = gradient_descent_reg(X, y, theta, alpha, lam, num_iters)
    predicted = predict(theta, X)
    return predicted, theta, costs

if __name__ == '__main__':
    images_dir = os.path.join(sys.path[0], 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    data = np.loadtxt(os.path.join(sys.path[0], 'logistic_regression_data.csv'), delimiter = ',', dtype = np.float64)
    X, y = data[:, :-1], data[:, -1].reshape((-1, 1))

    # overfitting without regularization
    power, num_iters = 10, 100000
    predicted, theta, costs = logistic_regression_reg(X, y, power = power, alpha = 0.6, lam = 0, num_iters = num_iters)
    print('The accuracy is {:.2f} %'.format(sum(predicted == y.flatten())/len(y)*100))

    u = np.linspace(min(X[:, 0]),max(X[:, 0]), 50)
    v = np.linspace(min(X[:, 1]),max(X[:, 1]), 50)

    z = np.zeros((len(u),len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(expand_feature(u[i].reshape(1,-1),v[j].reshape(1,-1), power = power),theta)
    z = np.transpose(z)

    plt.contour(u,v,z,[0,0.01], cmap = "Reds")
    sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = y.flatten())
    plt.title('Overfitting')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(images_dir, 'decision_boundary_overfitting.png'))
    plt.clf()

    # underfitting with too much regularization
    predicted, theta, costs = logistic_regression_reg(X, y, power = power, alpha = 0.6, lam = 5, num_iters = num_iters)
    print('The accuracy is {:.2f} %'.format(sum(predicted == y.flatten())/len(y)*100))

    z = np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(expand_feature(u[i].reshape(1,-1),v[j].reshape(1,-1), power = power),theta)
    z = np.transpose(z)

    plt.contour(u,v,z,[0,0.01], cmap = "Reds")
    sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = y.flatten())
    plt.title('Underfitting')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(images_dir, 'decision_boundary_underfitting.png'))
    plt.clf()

    # proper regularization
    predicted, theta, costs = logistic_regression_reg(X, y, power = power, alpha = 0.6, lam = 0.5, num_iters = num_iters)
    print('The accuracy is {:.2f} %'.format(sum(predicted == y.flatten())/len(y)*100))

    z = np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(expand_feature(u[i].reshape(1,-1),v[j].reshape(1,-1), power = power),theta)
    z = np.transpose(z)

    plt.contour(u,v,z,[0,0.01], cmap = "Reds")
    sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = y.flatten())
    plt.title('Adequate Regularization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(images_dir, 'decision_boundary_regularization.png'))
    plt.clf()
