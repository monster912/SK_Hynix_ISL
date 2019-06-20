import numpy as np
import matplotlib.pyplot as plt

# 정규분포를 따르는 합성데이터 생성함수 정의
def generate_normal(n_samples, train_test_ratio=0.8, seed=2019):
    np.random.seed(seed)
    n = n_samples // 2
    n_train = int(n * train_test_ratio)
    X1 = np.random.normal(loc=10, scale=5, size=(n, 2))
    X2 = np.random.normal(loc=20, scale=5, size=(n, 2))
    Y1 = np.ones(n)
    Y2 = - np.ones(n)
    X_train = np.concatenate((X1[:n_train], X2[:n_train]))
    X_test = np.concatenate((X1[n_train:], X2[n_train:]))
    Y_train = np.concatenate((Y1[:n_train], Y2[:n_train]))
    Y_test = np.concatenate((Y1[n_train:], Y2[n_train:]))
    return (X_train, Y_train), (X_test, Y_test)


# 데이터 플롯 함수 정의
def plot(data, labels, title='Train data'):
    plt.scatter(data[labels==1][:, 0], data[labels==1][:, 1], color='b', edgecolor='k', label='label : 1')
    plt.scatter(data[labels==-1][:, 0], data[labels==-1][:, 1], color='r', edgecolor='k', label='label : -1')
    plt.axvline(x=0, color='k')
    plt.axhline(y=0, color='k')
    plt.grid(True)
    plt.title(title)
    plt.legend()
    
    
    # Decision boundary를 그리는 함수정의
# meshgrid 메소드이용
def decision_boundary(w, xlim, ylim, colormap, bias_flag=False):
    xmin, xmax = xlim
    ymin, ymax = ylim
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
    grids = np.c_[xx.ravel(), yy.ravel()]
    if bias_flag:
        grids = add_bias(grids)
    pred = predict(w, grids)
    Z = pred.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=0, colors='k')
    if colormap == True:
        plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.7)
    
def draw_boundary(w, data, labels, title='Train data', colormap=False):
    # 먼저 데이터 플롯한다
    plot(data, labels, title=title)
    axes = plt.gca() # 현재 플롯된 axes객체를 가져온다
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    # 학습모델의 Decision boundary
    bias_flag = False
    if len(data.T) != len(w):
        bias_flag = True
    decision_boundary(w, xlim, ylim, colormap, bias_flag)
    
def predict(w, X):
    pred = X.dot(w)
    return pred

def add_bias(X):
    X_bias = np.concatenate((X, np.ones((len(X), 1))), axis=1)
    return X_bias