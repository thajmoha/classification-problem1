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
#_______________________________________________________________________________________________________________________________________________________________________

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

# simple logistic regression
# Train the logistic regression classifier
try_logistic_regression = 0
if try_logistic_regression == 1:
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)

# Plot the decision boundary for logistic regression
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")
    plt.show()

# Print accuracy
    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
          '% ' + "(percentage of correctly labelled datapoints)")

#________________________________________________________________________________________________________________________________________________________

# Neural network structure
n_h = 4
(n_x, n_h, n_y) = NN_structure(X, Y, n_h)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))
#___________________________________________________________________________________________________________________________________________________________

# initializing neural network parameters
n_x, n_h, n_o = initialize_parameters_test_case()

parameters = initial_param(n_x, n_h, n_o)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
#__________________________________________________________________________________________________________________________________________________________

A2, cache = forward_propagate(X, parameters)


print("cost = " + str(compute_cost(A2, Y)))
#__________________________________________________________________________________________________________________________________________________________

grads = back_propagate(parameters, cache, X, Y)

#___________________________________________________________________________________________________________________________________________________________

parameters = update_parameter(parameters, grads)


num_epochs = 10000
n_h = 4
# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h, num_epochs, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()


# Print accuracy
predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
