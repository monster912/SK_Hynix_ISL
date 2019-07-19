# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

# 정규분포를 따르는 합성데이터 생성함수 정의
def generate_normal(n_samples, train_test_ratio=0.8, val_portion=0, seed=2019):
    np.random.seed(seed)
    n = n_samples // 2
    n_train = int(n * train_test_ratio)
    X1 = np.random.normal(loc=10, scale=5, size=(n, 2))
    X2 = np.random.normal(loc=20, scale=5, size=(n, 2))
    Y2 = np.ones(n)
    Y1 = np.zeros(n)
    X_test = np.concatenate((X1[n_train:], X2[n_train:]))
    Y_test = np.concatenate((Y1[n_train:], Y2[n_train:]))
    if val_portion:
        n_val = int(n_train * val_portion)
        X_val = np.concatenate((X1[:n_val], X2[:n_val]))
        Y_val = np.concatenate((Y1[:n_val], Y2[:n_val])) 
        X_train = np.concatenate((X1[n_val :n_train], X2[n_val :n_train]))
        Y_train = np.concatenate((Y1[n_val :n_train], Y2[n_val :n_train]))
        return (X_train.T, Y_train), (X_val.T, Y_val), (X_test.T, Y_test)
    
    X_train = np.concatenate((X1[:n_train], X2[:n_train]))
    Y_train = np.concatenate((Y1[:n_train], Y2[:n_train]))
    return (X_train.T, Y_train), (X_test.T, Y_test)


# 데이터 플롯 함수 정의
def plot(data, labels, title='Train data', s=35, axis=False, xlim=None, ylim=None):
    plt.scatter(data.T[labels==1][:, 0], data.T[labels==1][:, 1], color='r', edgecolor='k', label='label : 1', s=s)
    plt.scatter(data.T[labels==0][:, 0], data.T[labels==0][:, 1], color='b', edgecolor='k', label='label : 0', s=s)
    plt.grid(True)
    plt.title(title)
    plt.legend()
    if axis:
        plt.axvline(x=0, color='black', linewidth=1)
        plt.axhline(y=0, color='black', linewidth=1)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    return None


# Decision boundary를 그리는 함수정의
# meshgrid 메소드이용
def decision_boundary(w, b, xlim, ylim, colormap):
    xmin, xmax = xlim
    ymin, ymax = ylim
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
    grids = np.c_[xx.ravel(), yy.ravel()]
    predict, _ = forward(w, b, grids.T, None)
    Z = predict.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='k')
    if colormap == True:
        plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.7)
    return None


def draw_boundary(w, b, data, labels, title='Train data', colormap=False, s=35, axis=False, xlim=None, ylim=None):
    # 먼저 데이터 플롯한다
    plot(data, labels, title=title, s=s, axis=axis, xlim=xlim, ylim=ylim)
    axes = plt.gca() # 현재 플롯된 axes객체를 가져온다
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    # 학습모델의 Decision boundary
    decision_boundary(w, b, xlim, ylim, colormap)
    return None
    
def sigmoid(z):
    '''
    Compute the sigmoid of z
    
    Arguments: A scalar or numpy array of any size
    '''
    return 1 / (1 + np.exp(-z))



def initialize_weights(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    ### manual seed
    np.random.seed(0)
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.random.randn(dim, 1)
    b = np.random.randn(1).item()
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b



def forward(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Return:
    yhat -- prediction of corresponding input
    
    Tips:
    - Write your code step by step for the propagation
    """
    
    m = X.shape[1]
    eps = 1e-8
    # FORWARD PROPAGATION
    ### START CODE HERE ### (≈ 2 lines of code)
    Yhat = sigmoid(np.dot(w.T, X) + b)
    cost = None
    if isinstance(Y, np.ndarray):
        cost = (- 1 / m) * np.sum(Y * np.log(Yhat + eps) + (1 - Y) * (np.log(1 - Yhat + eps)))
    ### END CODE HERE ###
    
    return Yhat, cost


def backward(w, b, X, Y, Yhat):
    
    '''
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    Yhat -- predicted label which can be interpreted as probability(= confidence)

    Return:
    grads -- gradient of parameters
    '''
    
    m = X.shape[1]
    # print(w, b, X, Y, Yhat)
    
    # BACKWARD PROPAGATION
    ### START CODE HERE ### (≈ 2 lines of code)
    # print(Yhat-Y)
    dw = (1 / m) * np.dot(X, (Yhat - Y).T)
    db = (1 / m) * np.sum(Yhat - Y)
    # print(dw, db)
    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    
    grads = {"dw": dw,
             "db": db}
    
    return grads

def fit(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use forward(), backward().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in tqdm_notebook(range(num_iterations)):
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        Yhat, cost = forward(w, b, X, Y)
        grads = backward(w, b, X, Y, Yhat)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # print(dw, db)
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
            # print(dw, db)
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

# GRADED FUNCTION: predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    assert(w.shape[0] == X.shape[0])
    
    ### START CODE HERE ### (≈ 4 line of code)
    Yhat = sigmoid(np.dot(w.T, X) + b)
    Yhat[Yhat > 0.5] = 1
    Yhat[Yhat <= 0.5] = 0
    Y_prediction = Yhat
    ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

# GRADED FUNCTION: model

def Logistic_regression(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_weights(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = fit(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    train_acc = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_acc = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    print("train accuracy: {} %".format(train_acc))
    print("test accuracy: {} %".format(test_acc))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations,
        "train_acc":train_acc,
        "test_acc":test_acc}
    
    return d
