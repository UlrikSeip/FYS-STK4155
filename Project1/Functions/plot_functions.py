import os
import sys
import pytest
import numba
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import functools
import time
from numba import jit
from PIL import Image
from sklearn.model_selection import train_test_split

# Add the src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'

sys.path.append(os.path.join(os.path.dirname(__file__), 'Functions'))
from LinearReg import linregOwn, linregSKL
from Franke import franke
from designMat import designMatrix
from Bootstrap import Bootstrap

######## a) Plot Franke Function without/with Noise ########

def plot_franke(x, y, noise = False, scalor = 0.05, method = None, seed1 = 8172, seed2 = 1726, absolute_error = False):
    x,y = np.meshgrid(x,y)
    print(x.shape)
    print(y.shape)
    f = franke(x,y) 
    print(f.shape)
    
    if(noise):
        f = franke(x,y) + scalor*np.random.normal(0, 1, franke(x,y).shape)
    
    if method == method:
        np.random.seed(seed1)
        x_new = np.random.rand(100)
        np.random.seed(seed2)
        y_new = np.random.rand(100)
        xn = x_new.ravel()
        yn = y_new.ravel()
        fn = franke(xn, yn)
        X = designMatrix(xn, yn)
        linreg = linregOwn(method = method)
        beta = linreg.fit(X, fn)
        
        xnew = np.linspace(0, 1, np.size(x_new))
        ynew = np.linspace(0, 1, np.size(x_new))
        Xnew, Ynew = np.meshgrid(xnew, ynew)
        F_true = franke(Xnew, Ynew)

        xn = Xnew.ravel()
        yn = Ynew.ravel()
        xb_new = designMatrix(xn, yn)
        f_predict = xb_new.dot(beta)
        F_predict = f_predict.reshape(F_true.shape)
    
    #Plot the Franke Function
    fig = plt.figure()
    ax = fig.gca(projection='3d') ##get current axis
    surf = ax.plot_surface(x, y, f, cmap= 'coolwarm', linewidth= 0, antialiased= False) ## colormap is coolwarm,
    ## antialiased controls the transparency of the surface
    if method == 'ols':
        surf = ax.plot_surface(Xnew, Ynew, F_predict, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
        if absolute_error == True:
            surf = ax.plot_surface(Xnew, Ynew, abs(F_predict-F_true), cmap = cm.coolwarm, linewidth = 0, antialiased = False)

    #Customize z axis
    ax.set_zlim(-0.10, 1.4)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(30, 45)
    #Labeling axes and title
    ax.set_title('Franke function without noise')
    if(noise):
        ax.set_title('Franke function with noise')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    #Add colour bar
    fig.colorbar(surf, shrink= 0.5, aspect= 0.5)
    return plt.show()
    



#plot_franke(x, y)
# plot_franke_noise(x, y)

##### Plot MSE vs complexity of the data for the train and test data sets

def plot_MSE_Complexity(x1, x2, y, random_state = None, ndegree = 5, graph = False):
    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, random_state = None)
    MSE_train     = []
    MSE_test      = []
    for degree in range(1, ndegree+1):
        X_train_     = designMatrix(x1_train, x2_train, degree)
        X_test_      = designMatrix(x1_test, x2_test, degree)
        linreg = linregOwn()
        beta_ = linreg.fit(X_train_, y_train)
        pred_train = linreg.predict(X_train_)
        MSE_train_ = linreg.MSE(y_train)
        MSE_train.append(MSE_train_)
    
        pred_test_ = linreg.predict(X_test_)
        MSE_test_ = linreg.MSE(y_test)
        MSE_test.append(MSE_test_)
    print('-------------------------------------------------')
    print('MSE_test: {}' .format(np.round(MSE_test, 4)))
    print('MSE_train: {}' .format(np.round(MSE_train, 4)))
    print('The polynomial fit of degree {} performs best' .format(MSE_test.index(min(MSE_test))+1))
    print('-------------------------------------------------')
    if graph == True:
        plot, ax = plt.subplots()
        plt.xlabel('Complexity (Order of polynomial)')
        plt.ylabel('MSE')
        plt.title('Change in MSE depending on the complexity of the model')
        plt.plot(range(1, ndegree+1), np.round(MSE_train, 4), label = 'Training Sample') 
        plt.plot(range(1, ndegree+1), np.round((MSE_test), 4), label = 'Test Sample')
        ax.axis([1,ndegree, 0, max(MSE_test)]) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.legend()
    return plt.show()


######## Bias-Variance-Tradeoff #############  

def plot_bias_var_tradeoff(x1, x2, y,ndegree = 5,  graph = True, method = 'ols'):
    bias = []
    var = []
    MSE = []
    for degree in range(1, ndegree + 1):
        bias_, var_, mse_ = Bootstrap(x1, x2, y, degree = degree, method = method)
        bias.append(bias_)
        var.append(var_)
        MSE.append(mse_)
    if graph == True:
        plot, ax = plt.subplots()
        plt.xlabel('Complexity (Order of polynomial)')
        plt.ylabel('MSE')
        plt.title('Bias-Variance tradeoff depending on the complexity of the model' )
        plt.plot(range(1, ndegree+1), np.round(MSE, 4), label = 'MSE') 
        plt.plot(range(1, ndegree+1), np.round(bias, 4), label= 'Bias')
        plt.plot(range(1, ndegree+1), np.round(var, 4), label = 'Variance') 
        ax.axis([1,ndegree, 0, np.max(MSE)+0.1]) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.legend()
        return plt.show()
    
