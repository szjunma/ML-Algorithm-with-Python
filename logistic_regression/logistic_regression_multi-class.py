import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from logistic_regression_reg import *

# prediction function is different from binary classification
def predict_multi_class(theta, X):
    return np.argmax(sigmoid(np.dot(X, theta)), axis = 1)

def logistic_regression_reg_multi_class(X, y, power = 2, alpha = 0.01, lam = 0, num_iters = 100):
    X = expand_feature(X[:, 0], X[:, 1], power = power)
    theta = np.zeros((X.shape[1], y.shape[1]), dtype = np.float64)
    theta, costs = gradient_descent_reg(X, y, theta, alpha, lam, num_iters)
    predicted = predict_multi_class(theta, X)
    return predicted, theta, costs

if __name__ == '__main__':
    images_dir = os.path.join(sys.path[0], 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    N = 80 # number of points per class
    D = 2 # dimensionality, we use 2D data for easy visulization
    K = 3 # number of classes, binary for logistic regression
    X = np.zeros((N * K, D), dtype = float) # data matrix (each row = single example, can view as xy coordinates)
    y_ = np.zeros(N * K, dtype = int) # class labels for plotting
    y = np.zeros((N * K, K), dtype = int) # class labels for training


    for i in range(K):
        r = np.random.normal(i + 0.5, 0.3, (N, 1)) # radius
        t = np.linspace(0, np.pi * 2, N).reshape(N, 1)  # theta

        X[i * N:(i + 1) * N] = np.append(r * np.sin(t), r * np.cos(t), axis = 1)
        y_[i * N:(i + 1) * N] = i
        y[i * N:(i + 1) * N, i] = 1

    sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = y_, palette = sns.color_palette('deep', K), edgecolor = "none")
    plt.title('Dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(images_dir, 'data_multi-class.png'))
    plt.clf()

    num_iters = 2000
    predicted, theta, costs = logistic_regression_reg_multi_class(X, y, alpha = 0.3, lam = 0, num_iters = num_iters)
    print('The accuracy is {:.2f} %'.format(sum(predicted == y_)/len(y_)*100))

    gridsize = 200
    u = np.linspace(min(X[:, 0]),max(X[:, 0]), gridsize)
    v = np.linspace(min(X[:, 1]),max(X[:, 1]), gridsize)

    gridx, gridy = np.meshgrid(u, v)
    grid = np.array([gridx.reshape(-1, ), gridy.reshape(-1, )]).T

    z = predict_multi_class(theta, expand_feature(gridx.reshape(-1, 1), gridy.reshape(-1, 1))).reshape(gridsize, gridsize)
    plt.contourf(u, v, z, alpha = 0.2, levels = K - 1, antialiased = True)

    sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = y_, palette = sns.color_palette('deep', K), edgecolor = "none")
    plt.title('Decision Boundary')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(images_dir, 'decision_boundary_multi-class.png'))
    plt.clf()
