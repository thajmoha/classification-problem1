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
from code2_gen_planner_data import load_planar_dataset
np.random.seed(22)  # set a seed so that the results are consistent --- np.random.seed(22)  and rng(22) in MATLAB produce same results
#____________________________________________________________________________________________________________________________________________________________

# generating data
X, Y = load_planar_dataset()
print('shape of X:', X.shape)
print('shape of Y:', Y.shape)
m = X.shape[1]
print('number of training examples:', m)

plt.figure()
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral);
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()
#____________________________________________________________________________________________________________________________________________________________

# This may take about 2 minutes to run

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_epochs=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
plt.show()
