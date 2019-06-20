import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from math import sqrt



def load_planar_dataset(n_samples, noise=0.2):
    np.random.seed(1)
    m = n_samples # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros(m, dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*noise # theta
        r = a*np.sin(4*t) + np.random.randn(N)*noise # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    return X, Y

def plot(data, labels, title='Train data', s=35, axis=False, xlim=None, ylim=None):
    plt.scatter(data.T[labels==1][:, 0], data.T[labels==1][:, 1], color='b', edgecolor='k', label='label : 1', s=s)
    plt.scatter(data.T[labels==0][:, 0], data.T[labels==0][:, 1], color='r', edgecolor='k', label='label : 0', s=s)
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
    
def decision_boundary(model, xlim, ylim, colormap):
    xmin, xmax = xlim
    ymin, ymax = ylim
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
    grids = np.c_[xx.ravel(), yy.ravel()]
    if model.model_name == 'LS':
        predict = model.predict(grids)
    else:
        predict = model.predict(grids.T)
    Z = predict.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=0.5, colors='k')
    if colormap == True:
        plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.6)
        
def draw_boundary(data, labels, model, title='Train data', colormap=False, s=35, axis=False, xlim=None, ylim=None):
    # 먼저 데이터 플롯한다
    plot(data, labels, title=title, s=s, axis=axis, xlim=xlim, ylim=ylim)
    axes = plt.gca() # 현재 플롯된 axes객체를 가져온다
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    # 학습모델의 Decision boundary
    decision_boundary(model, xlim, ylim, colormap)
    
   # 지난 시간에 구현했던 모델사용 : Least Square
class LeastSquare:
    def __init__(self, data, labels, bias=False, weight_decay=0):
        if bias == True:
            self.data = self.add_bias(data.T)
        else:
            self.data = data.T
        self.labels = labels.reshape(-1, 1)
        self.add_reg(weight_decay)
        self.bias = bias
        self.model_name='LS'
        
    def fit(self):
        self.w = (np.linalg.inv(self.data.T@self.data)@self.data.T)@self.labels
        
    def predict(self, data):
        if self.bias == True:
            data = self.add_bias(data)
        return data.dot(self.w)
    
    def get_accuracy(self, data, labels):
        z = self.predict(data)
        yhat = np.sign(z)
        n_samples = len(labels)
        n_correct = (yhat == labels.reshape(-1, 1)).sum()
        acc = n_correct / n_samples * 100
        return acc
    
    def add_bias(self, data):
        return np.concatenate((data, np.ones((len(data), 1))), axis=1)
    
    def add_reg(self, weight_decay):
        weight_decay = sqrt(weight_decay)
        add_data = np.diag([weight_decay] * len(self.data.T))
        add_labels = np.zeros((len(self.data.T), 1))
        self.data = np.concatenate((self.data, add_data))
        self.labels = np.concatenate((self.labels, add_labels))
        
# 지난 시간에 구현했던 모델사용 : Logsitic Regression
class Logistic:
    def __init__(self, data, labels, seed=2019):
        np.random.seed(seed)
        self.w = np.random.randn(len(data), 1)
        self.b = np.random.randn(1, 1)
        self.data = data
        self.labels = labels
        self.model_name='LR'
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _forward(self):
        z = self.w.T.dot(self.data) + self.b
        self.yhat = self.sigmoid(z)
        return self.yhat
    
    def _backward(self):
        self.dw = (self.data * (self.yhat - self.labels)).sum(axis=1, keepdims=True) / len(self.labels)
        self.db = (self.yhat - self.labels).sum() / len(self.labels)
        
    def fit(self, iteration, lr, verbose=False):
        for i in range(iteration):
            self._forward()
            self._backward()
            self.w -= lr * self.dw
            self.b -= lr * self.db
            if verbose:
                train_acc = self.get_accuracy(self.data, self.labels)
                print('[{}/{}] iterations'.format(i+1, iteration))
                print('Train accuracy: {:.4f}'.format(train_acc))
            
    def predict(self, data):
        z = self.w.T.dot(data) + self.b
        yhat = self.sigmoid(z)
        return yhat
    
    def get_accuracy(self, data, labels):
        yhat = self.predict(data)
        decision_0 = yhat < 0.5
        decision_1 = yhat >= 0.5
        yhat[decision_0] = 0
        yhat[decision_1] = 1
        n_correct = (yhat == labels).sum()
        acc = n_correct / len(labels) * 100
        return acc