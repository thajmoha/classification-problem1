# main code for neural network
# Turki Haj Mohamad
# 11/18/2018
#_______________________________________________________________________________________________________________________________________________________________________

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from code1_NN_functions import plot_decision_boundary, sigmoid, NN_structure, initial_param, forward_propagate, compute_cost, back_propagate, update_parameter, nn_model, predict
from code2_gen_planner_data import load_extra_datasets

np.random.seed(22)  # set a seed so that the results are consistent --- np.random.seed(22)  and rng(22) in MATLAB produce same results
#____________________________________________________________________________________________________________________________________________________________

# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

# START CODE HERE ### (choose your dataset)
dataset = "noisy_moons"
### END CODE HERE ###

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y % 2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral)
plt.grid()
plt.show()


# This may take about 2 minutes to run

hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    #plt.subplot(5, 2, i + 1)
    plt.figure(i)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_epochs=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
plt.show()
