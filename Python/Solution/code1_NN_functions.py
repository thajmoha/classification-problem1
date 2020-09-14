# NN functions
# Turki Haj Mohamad
# 11/18/2018
#_______________________________________________________________________________________________________________________________________________________________________

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

np.random.seed(22)


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)


def NN_structure(X, Y, n_h):
    """
    find the neural network structure
    Argument: X (matrix) [num_feat x num_examples]
            Y (matrix) [num_output x num_examples]
            n_h (scalar) number of hidden units
    return: n_x (scalar) number of featurs (input)
        n_h (scalar) number of hidden units
        n_o (scalar) number of output
    """
    n_x = X.shape[0]
    n_h = n_h
    n_o = Y.shape[0]
    return n_x, n_h, n_o


def initial_param(n_x, n_h, n_o):
    """
    initalizing parameters for neural network
    Argument: n_x (scalar) number of featurs (input)
        n_h (scalar) number of hidden units
        n_o (scalar) number of output
     retuen: w1 (matrix)
            b1 (vector)
            w2 (matrix)
            b2  (vector)
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_o, n_h) * 0.01
    b2 = np.zeros((n_o, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_o, n_h))
    assert (b2.shape == (n_o, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagate(X, parameters):
    """
    computing forward propagation for neual network
    Argument: X (matrix) input data of size (n_x, m)
    parameters  (python dictionary) containing your parameters (output of initialization function)

    Returns:
    A2 -- (matrix) [n_o x m] The sigmoid output of the second activation
    cache -- (Python dictionary) containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache

def compute_cost(A2,Y):
    loss = -(Y*np.log(A2)+(1-Y)*np.log(1-A2))
    cost = np.mean(loss)

    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.
                                # E.g., turns [[17]] into 17
    assert(isinstance(cost, float))

    return cost



def back_propagate(parameters,cache,X,Y):
    A1 = cache["A1"]
    A2 = cache["A2"]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    m = X.shape[1]

    dZ2 = A2 - Y
    dW2 = 1/m*np.dot(dZ2,A1.T)
    db2 = np.mean(dZ2,axis=1,keepdims=True)
    #dZ1 = np.dot(W2.T,dZ2)*(A1*(1-A1)) #in case of sigmoid as hidden unit
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = 1/m*np.dot(dZ1,X.T)
    db1 = np.mean(dZ1,axis=1,keepdims=True)

    grads = {"dW1":dW1,
    "db1":db1,
    "dW2":dW2,
    "db2":db2}

    return grads

def update_parameter(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    W2 = W2 - learning_rate*dW2
    b1 = b1 - learning_rate*db1
    b2 = b2 - learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X,Y,n_h, num_epochs = 10000, print_cost = False):


    (n_x, n_h, n_o) = NN_structure(X, Y, n_h)
    parameters = initial_param(n_x, n_h, n_o)

    for i in range(num_epochs):
        A2, cache = forward_propagate(X, parameters)
        cost = compute_cost(A2, Y)
        grads = back_propagate(parameters, cache, X, Y)
        parameters = update_parameter(parameters, grads)
        if print_cost and i%1000 ==0:
            print("Cost after iteration %i: %f" %(i,cost))
    return parameters


def predict(parameters, X):
    A2, cache = forward_propagate(X, parameters)
    predictions = A2>0.5
    return predictions







