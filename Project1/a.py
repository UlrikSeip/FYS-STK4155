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
from sklearn.preprocessing import StandardScaler

# Add the Functions/ directory to the python path so we can import the code 

sys.path.append(os.path.join(os.path.dirname(__file__), 'Functions'))
from LinearReg import linregOwn, linregSKL
from Franke import franke
from designMat import designMatrix
from Bootstrap import Bootstrap

######## a) Plot Franke Function without/with Noise ########

def plot_franke(x, y, franke_ = False, noise = False, scalor = 0.05, method = 'ols', seed1 = 8172, lambda_ = 0.005, absolute_error = False):
    """
    Plots the franke function.Franke's function has two Gaussian peaks of different heights, 
    and a smaller dip. It is used as a test function in interpolation problems.
    The function is evaluated on the square xi ∈ [0, 1], for all i = 1, 2.
    
    Reference: Franke, R. (1979). A critical comparison of some methods for interpolation of 
    scattered data (No. NPS53-79-003). NAVAL POSTGRADUATE SCHOOL MONTEREY CA.
    
    
    Arguments:
    x:  1-dimensional numpy array (1D np.array)
    y:  1-dimensional numpy array (1D np.array)
    franke_: binary argument with inputs True/False. If 'True', plots the franke function
    noise: binary argument with inputs True/False. If 'True', plots the franke function with added noise. 
           Activated only when franke_ == True.
    scalor: float type,  controls the amount of noise to be added to the franke function. Activated only when
            noise == True.
    method: character input accepting 'ols', 'ridge', 'lasso'. Plots the corresponding model fit.
    seed1: float type. used for reproducable output
    lambda_: float type. Activated only when method = 'ridge' or 'lasso'. Controls the amount of shrinkage of the
             parameters. Higher number indicates higher shrinkage.
    absolute_error: Binary type with inputs True/False. If 'True', outputs a plot of absolute deviation of the true
                    franke values and the fit of the corresponding model. Activated only when method is either 'ols',
                    'ridge' or 'lasso
    """
    x,y = np.meshgrid(x,y)
    f = franke(x,y) ##true franke values
    
    if(noise):  ##noisy franke values
        f = franke(x,y) + scalor*np.random.normal(0, 1, franke(x,y).shape)
    
    if method == 'ols': ##fit and predict ols
        np.random.seed(seed1)
        x_new = np.random.rand(500)
        y_new = np.random.rand(500)
        xn = x_new.ravel()
        yn = y_new.ravel()
        fn = franke(xn, yn) 
        X = designMatrix(xn, yn)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X[:, 0] = 1
        linreg = linregOwn(method = 'ols')
        beta = linreg.fit(X, fn)
        
        xnew = np.linspace(0, 1, np.size(x_new))
        ynew = np.linspace(0, 1, np.size(x_new))
        Xnew, Ynew = np.meshgrid(xnew, ynew)
        F_true = franke(Xnew, Ynew)

        xn = Xnew.ravel()
        yn = Ynew.ravel()
        xb_new = designMatrix(xn, yn)
        scaler = StandardScaler()
        scaler.fit(xb_new)
        xb_new = scaler.transform(xb_new)
        xb_new[:, 0] = 1
    
        f_predict = np.dot(xb_new, beta)
        F_predict = f_predict.reshape(F_true.shape)
    
    if method == 'ridge': ##fit and predict ridge
        np.random.seed(seed1)
        x_new = np.random.rand(500)
        y_new = np.random.rand(500)
        xn = x_new.ravel()
        yn = y_new.ravel()
        fn = franke(xn, yn) 
        X = designMatrix(xn, yn)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X[:, 0] = 1
        linreg = linregOwn(method = 'ridge')
        beta = linreg.fit(X, fn, lambda_ = 0.1)
        
        xnew = np.linspace(0, 1, np.size(x_new))
        ynew = np.linspace(0, 1, np.size(x_new))
        Xnew, Ynew = np.meshgrid(xnew, ynew)
        F_true = franke(Xnew, Ynew)

        xn = Xnew.ravel()
        yn = Ynew.ravel()
        xb_new = designMatrix(xn, yn)
        scaler = StandardScaler()
        scaler.fit(xb_new)
        xb_new = scaler.transform(xb_new)
        xb_new[:, 0] = 1
    
        f_predict = np.dot(xb_new, beta)
        F_predict = f_predict.reshape(F_true.shape)
    
    if method == 'lasso': ##fit and predict lasso
        np.random.seed(seed1)
        x_new = np.random.rand(500)
        y_new = np.random.rand(500)
        xn = x_new.ravel()
        yn = y_new.ravel()
        fn = franke(xn, yn) 
        X = designMatrix(xn, yn)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X[:, 0] = 1
        linreg = linregOwn(method = 'lasso')
        beta = linreg.fit(X, fn, lambda_ = 0.1)
        
        xnew = np.linspace(0, 1, np.size(x_new))
        ynew = np.linspace(0, 1, np.size(x_new))
        Xnew, Ynew = np.meshgrid(xnew, ynew)
        F_true = franke(Xnew, Ynew)

        xn = Xnew.ravel()
        yn = Ynew.ravel()
        xb_new = designMatrix(xn, yn)
        scaler = StandardScaler()
        scaler.fit(xb_new)
        xb_new = scaler.transform(xb_new)
        xb_new[:, 0] = 1
    
        f_predict = np.dot(xb_new, beta)
        F_predict = f_predict.reshape(F_true.shape)
        
    #Plot the Franke Function
    fig = plt.figure()
    ax = fig.gca(projection='3d') ##get current axis
    ## antialiased controls the transparency of the surface
    
    if method == 'ols':
        if absolute_error == True:
            surf = ax.plot_surface(Xnew, Ynew,abs(F_predict-F_true), cmap = cm.coolwarm, linewidth = 0, antialiased = False)
        else:
            surf = ax.plot_surface(Xnew, Ynew,F_predict, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
    if method == 'ridge':
        if absolute_error == True:
            surf = ax.plot_surface(Xnew, Ynew,abs(F_predict-F_true), cmap = cm.coolwarm, linewidth = 0, antialiased = False)
        else:
            surf = ax.plot_surface(Xnew, Ynew,F_predict, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
    if method == 'lasso':
        if absolute_error == True:
            surf = ax.plot_surface(Xnew, Ynew,abs(F_predict-F_true), cmap = cm.coolwarm, linewidth = 0, antialiased = False)
        else:
            surf = ax.plot_surface(Xnew, Ynew,F_predict, cmap = cm.coolwarm, linewidth = 0, antialiased = False)

    #Customize z axis
    if franke_ == True:
        surf = ax.plot_surface(x, y, f, cmap= 'coolwarm', linewidth= 0, antialiased= False) ## colormap is coolwarm,
        ax.set_title('Franke function without noise')
        if(noise):
            ax.set_title('Franke function with noise')
            
    ax.set_zlim(-0.10, 1.4)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(30, 45)
    #Labeling axes and title

    if method == 'ols':
        ax.set_title('OLS Fit')
    if method == 'ridge':
        ax.set_title('Ridge Fit')
    if method == 'lasso':
        ax.set_title('Lasso Fit')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    #Add colour bar
    fig.colorbar(surf, shrink= 0.5, aspect= 0.5)
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'franke_abs_lasso.png'), transparent=True, bbox_inches='tight')
    return plt.show()
 
 
 
 
#### Plots ##############################################   
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

##Franke function no noise
plot_franke(x, y, franke_ = True, noise = False, method = None)

##Franke function with noise
plot_franke(x, y, franke_ = True, noise = True, method = None)

##OLS fit on franke function/ without noise
plot_franke(x, y, noise = False, method = 'ols', absolute_error = False)

##Ridge fit on franke function/ outcome variable does not contain noise
plot_franke(x, y, noise = False, method = 'ridge', absolute_error = False)

##Lasso fit on franke function / outcome variable does not contain noise
plot_franke(x, y, noise = False, method = 'lasso', absolute_error = False)


##OLS fit on franke function/ without noise
plot_franke(x, y, noise = False, method = 'ols', absolute_error = True)

##Ridge fit on franke function/ outcome variable does not contain noise
plot_franke(x, y, noise = False, method = 'ridge', absolute_error = True)

##Lasso fit on franke function / outcome variable does not contain noise
plot_franke(x, y, noise = False, method = 'lasso', absolute_error = True)




