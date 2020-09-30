import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from neural_network import *

if __name__ == '__main__':
    images_dir = os.path.join(sys.path[0], 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # load train and test data
    train = np.loadtxt(os.path.join(sys.path[0], 'data/train.csv'), delimiter = ',', skiprows = 1)
    test = np.loadtxt(os.path.join(sys.path[0], 'data/test.csv'), delimiter = ',', skiprows = 1)
    trainx, trainy_ = train[:, 1:], train[:, 0].flatten()
    trainx_norm = trainx/255

    # generate one-hot trainy
    trainy = np.zeros((trainx.shape[0], 10), dtype = int)
    for i, v in enumerate(trainy_):
        trainy[i, int(v)] = 1

    num_iters = 2000
    mnist_para = nnet(trainx_norm, trainy, step_size = 0.4, lam = 0.001, h = 10, num_iters = num_iters)
    predicted = predict(trainx_norm, mnist_para)
    print('The accuracy is {:.2f} %'.format(sum(predicted == trainy_)/len(trainy_)*100))
