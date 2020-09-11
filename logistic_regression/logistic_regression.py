import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cost(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    cos = -(np.sum(y * np.log(h)) + np.sum((1 - y) * np.log(1 - h)))/len(y)
    return cos

def expand_feature(x1, x2, power = 2):
    #expand a 2D feature matrix to polynimial features up to the power
    new_x = np.ones((x1.shape[0], 1))
    for i in range(1, power + 1):
        for j in range(i + 1):
            new_x = np.append(new_x, (x1**(i-j)*(x2**j)).reshape(-1, 1), axis = 1)
    return new_x

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    costs = []
    for _ in range(num_iters):
        h = sigmoid(np.dot(X, theta))
        theta -= alpha * np.dot(X.T, (h - y))/m
        costs.append(cost(theta, X, y))
    return theta, costs

def predict(theta, X):
    return (sigmoid(np.dot(X, theta)) > 0.5).flatten()

def logistic_regression(X, y, power = 2, alpha = 0.01, num_iters = 100):
    X = expand_feature(X[:, 0], X[:, 1], power = power)
    theta = np.zeros((X.shape[1], 1), dtype = np.float64)
    theta, costs = gradient_descent(X, y, theta, alpha, num_iters)
    predicted = predict(theta, X)
    return predicted, theta, costs

if __name__ == '__main__':
    images_dir = os.path.join(sys.path[0], 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    data = np.loadtxt(os.path.join(sys.path[0], 'logistic_regression_data.csv'), delimiter = ',', dtype = np.float64)
    X, y = data[:, :-1], data[:, -1].reshape((-1, 1))

    sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = y.flatten())
    plt.title('Dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(images_dir, 'data.png'))
    plt.clf()

    predicted, theta, costs = logistic_regression(X, y, alpha = 0.15, num_iters = 4000)
    print('The accuracy is {:.2f} %'.format(sum(predicted == y.flatten())/len(y)*100))

    sns.lineplot(range(4000), costs)
    plt.title('Cost vs Number of Interations')
    plt.ylabel('Cost')
    plt.xlabel('No. of Interations')
    plt.savefig(os.path.join(images_dir, 'cost.png'))
    plt.clf()

    u = np.linspace(min(X[:, 0]),max(X[:, 0]), 50)
    v = np.linspace(min(X[:, 1]),max(X[:, 1]), 50)

    z = np.zeros((len(u),len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(expand_feature(u[i].reshape(1,-1),v[j].reshape(1,-1)),theta)
    z = np.transpose(z)

    plt.contour(u,v,z,[0,0.01], cmap = "Reds")
    sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = y.flatten())
    plt.title('Decision Boundary')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(images_dir, 'decision_boundary.png'))
    plt.clf()
