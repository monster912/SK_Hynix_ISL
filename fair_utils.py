import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal # generating synthetic data


def generate_unfair_data(n_samples, disc_factor1, disc_factor2, p=0.8, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    
    # this determines the discrimination in the data - decraese it to generate more discrimination
    # disc_factor = math.pi / 4.0 (default)
    # disc_factor = math.pi / 8.0
    n = n_samples // 2
    num_train = int(n_samples * p)

    def generate_gaussian(mean, cov, label):
        pdf = multivariate_normal(mean=mean, cov=cov)
        X = pdf.rvs(n)
        Y = np.ones(n, dtype=float) * label
        return pdf, X, Y

    """Generate the non-sensitive features randomly"""
    # We will generate one gaussian cluster for each class
    mean1, sigma1 = [-5, -5], [[25, 0], [0, 25]]
    mean2, sigma2 = [5, 5], [[25, 0], [0, 25]]
    pdf1, X1, Y1 = generate_gaussian(mean1, sigma1, 0) # negative class (non-reoffend)
    pdf2, X2, Y2 = generate_gaussian(mean2, sigma2, 1) # positive class (reoffend)

    # join the posisitve and negative class clusters
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2))

    # shuffle the data
    perm = list(range(0, n_samples))
    random.shuffle(perm)
    X = X[perm]
    Y = Y[perm]
    
    rotation_matrix = disc_factor1 * np.array([[math.cos(disc_factor2), -math.sin(disc_factor2)],
                                               [math.sin(disc_factor2), math.cos(disc_factor2)]])
    X_aux = np.dot(Normalize(X), rotation_matrix)

    """Generate the sensitive feature here"""
    z_list = [] # this array holds the sensitive feature value
    for i in range (0, len(X)):
        x = X_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = pdf1.pdf(x)
        p2 = pdf2.pdf(x)
        
        # normalize the probabilities from 0 to 1
        s = p1 + p2
        p1 = p1 / s
        p2 = p2 / s
        
        r = np.random.uniform() # generate a random number from 0 to 1

        if r < p1: # the first cluster is the positive class
            z_list.append(1.0) # 1.0 means its black
        else:
            z_list.append(0.0) # 0.0 -> white

    z = np.array(z_list)
    
    X_train = X[:num_train]
    Y_train = Y[:num_train]
    z_train = z[:num_train]
    X_test = X[num_train:]
    Y_test = Y[num_train:]
    z_test = z[num_train:]
    return (X_train.T, Y_train, z_train), (X_test.T, Y_test, z_test)

def plot_unfair_data(X, Y, z, num_to_draw=200):
    X_draw = X[:num_to_draw]
    Y_draw = Y[:num_to_draw]
    z_draw = z[:num_to_draw]

    X_z_0 = X_draw[z_draw == 0.0]
    X_z_1 = X_draw[z_draw == 1.0]
    Y_z_0 = Y_draw[z_draw == 0.0]
    Y_z_1 = Y_draw[z_draw == 1.0]
    plt.scatter(X_z_0[Y_z_0==1.0][:, 0], X_z_0[Y_z_0==1.0][:, 1], color='red', marker='o', facecolors='none', s=90, linewidth=1.5, label="White reoffend")
    plt.scatter(X_z_0[Y_z_0==0.0][:, 0], X_z_0[Y_z_0==0.0][:, 1], color='blue', marker='o', facecolors='none', s=90, linewidth=1.5, label="White non-reoffend")
    plt.scatter(X_z_1[Y_z_1==1.0][:, 0], X_z_1[Y_z_1==1.0][:, 1], color='red', marker='o', facecolors='black', s=90, linewidth=1.5, label="Black reoffend")
    plt.scatter(X_z_1[Y_z_1==0.0][:, 0], X_z_1[Y_z_1==0.0][:, 1], color='blue', marker='o', facecolors='black', s=90, linewidth=1.5, label="Black non-reoffend")

    plt.tick_params(axis='x', which='both')#, bottom=False, top=False)#, labelbottom='off') # dont need the ticks to see the data distribution
    plt.tick_params(axis='y', which='both')#, left=False, right=False)#, labelleft='off')
    plt.legend(fontsize=16)
    return None

def decision_boundary(w, b, xlim, ylim, colormap):
    xmin, xmax = xlim
    ymin, ymax = ylim
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
    grids = np.c_[xx.ravel(), yy.ravel()]
    pred, _ = forward(w, b, grids.T)
    Z = pred.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5])
    if colormap == True:
        plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.9)
    return None

def draw_boundary(w, b, X, Y, z, num_to_draw=200, title='Train data', colormap=False, s=35, axis=False, xlim=None, ylim=None):
    if X.shape[0] > X.shape[1]:
        plot_unfair_data(X, Y, z, num_to_draw)
    else:
        plot_unfair_data(X.T, Y, z, num_to_draw)
    axes = plt.gca() # 현재 플롯된 axes객체를 가져온다
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    # 학습모델의 Decision boundary
    decision_boundary(w, b, xlim, ylim, colormap)
    return None

def add_line(xlim, ylim, w, b, level=0.0):
    slope = - w[0] / w[1]
    bias = - b / w[1]
    xrange = np.linspace(xlim[0], xlim[1], 1000)
    yrange = slope * xrange + bias
    
    plt.plot(xrange, yrange, "-", c=cm.binary(level), linewidth=1.0)
    return None

def forward(w, b, X, Y=None):
    m = X.shape[1]
    Yhat = sigmoid(np.dot(w.T, X) + b)
    cost = None
    if Y is not None:
        cost = (- 1 / m) * np.sum(Y * np.log(Yhat) + (1 - Y) * (np.log(1 - Yhat)))
    return Yhat, cost

def backward(w, b, X, Y, Yhat):
    m = X.shape[1]
    dw = (1 / m) * np.dot(X, (Yhat - Y).T)
    db = (1 / m) * np.sum(Yhat - Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    
    grads = {"dw": dw, "db": db}
    return grads

def fit(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation (≈ 1-4 lines of code)
        Yhat, cost = forward(w, b, X, Y)
        grads = backward(w, b, X, Y, Yhat)
        
        # Retrieve derivatives from grads
        dw, db = grads["dw"], grads["db"]
        
        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and (i + 1) % 100 == 0 or (i + 1) == num_iterations:
            print ("Cost after iteration [%i/%i]: %f" % (i + 1, num_iterations, cost), end='\r')
    print()
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    assert(w.shape[0] == X.shape[0])
    
    Yhat = sigmoid(np.dot(w.T, X) + b)
    Yhat[Yhat > 0.5] = 1
    Yhat[Yhat <= 0.5] = 0
    Y_prediction = Yhat
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def Logistic(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = fit(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w, b = parameters["w"], parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test accuracies
    train_acc = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_acc = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    print("train accuracy: {} %".format(train_acc))
    print("test accuracy: {} %".format(test_acc))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train,
         "train_acc": train_acc,
         "test_acc": test_acc,
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

def Normalize(data):
    temp = np.array(data)
    if data.shape[0] < data.shape[1]:
        temp = np.array(data).T
    mean, std = np.mean(temp, axis=0), np.std(temp, axis=0)
    temp = (temp - mean) / std
    if data.shape[0] < data.shape[1]:
        return temp.T
    return temp

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b