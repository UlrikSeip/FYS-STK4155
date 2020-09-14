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
from Crossvalidation import CrossValidation

##### Plot MSE vs complexity of the data for the train and test data sets
def plot_MSE_Complexity(degree = 7, graph = True):
    """
    Plots mean squared error (MSE) as a function of the complexity (i.e. polynomial degree) parameter 
    on the train and test data set. MSE is calculated using the OLS on the train and test data sets.
    
    Arguments:
    degree: integer type. complexity of the model (i.e. polynomial degree)
    graph: Binary type with inputs True/False. If True, plots the MSE on the train and test data
    """
    
    ##Make synthetic data
    n = 500
    np.random.seed(18271)
    x1 = np.random.rand(n)
    np.random.seed(91837)
    x2 = np.random.rand(n)
    y = franke(x1, x2) + 0.1*np.random.normal(0, 1, x1.size)
    
    ##split in train and test set
    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2)
    MSE_train     = []
    MSE_test      = []
    ##fit OLS with polynomial of different degree and compute MSE on the train and test data
    for degs in range(1, degree+1):
        X_train_     = designMatrix(x1_train, x2_train, degs)
        X_test_      = designMatrix(x1_test, x2_test, degs)
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
        plt.plot(range(1, degree+1), np.round(MSE_train, 4),'k--', label = 'Training Sample') 
        plt.plot(range(1, degree+1), np.round((MSE_test), 4), 'r-',label = 'Test Sample')
        ax.axis([1,degree, 0, max(MSE_test)]) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.legend()
        plt.subplots_adjust(left=0.2,bottom=0.2,right=0.9)
        #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'MSE_train_test.png'), transparent=True, bbox_inches='tight')
    return plt.show()


######## Bias-Variance-Tradeoff using either bootstrap or cross-validation #############  
def plot_bias_var_tradeoff(ndegree = 6, sampling_method = 'bootstrap', method = 'ols'):
    
    """
    Plots the bias-variance tradeoff using either bootstrap or cross-validation.
    
    Arguments:
    
    ndegree: integer type. Complexity of the model, degree of polynomial
    sampling_method: character type. Accepts only 'bootstrap' or 'cv'
    method:  character type. Accepts only the arguments 'ols', 'ridge', 'lasso'
    """
    
    ##Create synthetic data
    n = 500
    np.random.seed(18271)
    x1 = np.random.rand(n)
    np.random.seed(91837)
    x2 = np.random.rand(n)
    y = franke(x1, x2) + 0.1*np.random.normal(0, 1, x1.size)
    bias = []
    var = []
    MSE = []
    ##compute bias, variance and mse using bootstrap
    if sampling_method == 'bootstrap':
        for deg in range(1, ndegree + 1):
            bias_, var_, mse_, betavar_ = Bootstrap(x1, x2, y, degrees = deg, method = method)
            bias.append(bias_)
            var.append(var_)
            MSE.append(mse_)
    ##compute bias, variance, mse using cross-validation
    if sampling_method == 'cv':
        linreg = linregOwn(method=method)
        cv = CrossValidation(linreg, designMatrix)
        for deg in range(1, ndegree + 1):
            means =  cv.kFoldCV(x1, x2, y,10, degree = deg)
            bias.append(means[3])
            var.append(means[2])
            MSE.append(means[0])
            
    plot, ax = plt.subplots()
    plt.xlabel('Complexity (Order of polynomial)')
    plt.ylabel('MSE')
    if sampling_method == 'bootstrap':
        plt.title('Bias-Variance tradeoff using bootstrap' )
    if sampling_method == 'cv':
        plt.title('Bias-Variance tradeoff using cross-validation')
    plt.plot(range(1, ndegree+1), MSE, 'k-o', label = 'MSE') 
    plt.plot(range(1, ndegree+1), bias, 'b-o',label= 'Bias')
    plt.plot(range(1, ndegree+1), var, 'r-o',label = 'Variance') 
    #ax.axis([1,ndegree, 0, 1.1*np.max(var)]) 
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    plt.subplots_adjust(left=0.2,bottom=0.2,right=0.9)
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'ols_bias_var_cv.png'), transparent=True, bbox_inches='tight')
    return plt.show()
  
  
##train and test data MSE  
plot_MSE_Complexity(degree = 8, graph = True)  


##MSE bias-variance tradeoff using bootstrap
plot_bias_var_tradeoff(ndegree=6, sampling_method= 'bootstrap')

##MSE bias-variance tradeoff using cross-validation
plot_bias_var_tradeoff(ndegree=6, sampling_method = 'cv')       
       

def plot_franke_noise(method = 'ols') :
    
    """
    Plots mean squared error (MSE) and R2 score as a function of the noise scalor 
    (i.e. parameter controlling the amount of noise).
    
    Arguments:
    method: Character type. Activated only when method = 'ols', 'ridge', or 'lasso'. Else raises an error
    
    """
    ##Make synthetic data
    n = 1000
    np.random.seed(18271)
    x1 = np.random.rand(n)
    np.random.seed(91837)
    x2 = np.random.rand(n)
    y = franke(x1, x2) + 0.1*np.random.normal(0, 1, n)
    
    R2           = []
    MSE          = []
    R2_noise           = []
    MSE_noise          = []

    noise = np.logspace(-4,0,50)
    k = 1

    
    for eta in noise :
        y_data_noise = y +  eta * np.random.standard_normal(size = y.size)
        linreg = linregOwn(method = method)
        x1_train, x1_test, x2_train, x2_test, y_train, y_test, y_noise_train, y_noise_test = train_test_split(x1, x2, y,y_data_noise, test_size=0.35, random_state = 42)
        X_train = designMatrix(x1_train, x2_train, 3)
        X_test = designMatrix(x1_test, x2_test, 3)
        linreg.fit(X_train, y_noise_train)
        linreg.predict(X_test)
        MSE_noise.         append(linreg.MSE(y_noise_test))
        R2_noise.          append(linreg.R2(y_noise_test))
        
        linreg_NOnoise = linregOwn()
        linreg_NOnoise.fit(X_train, y_train)
        linreg_NOnoise.predict(X_test)
        MSE.append(linreg.MSE(y_test))
        R2. append(linreg.R2(y_test))
    
    print(1-np.array(R2_noise))
    fig, ax1 = plt.subplots()
    ax1.loglog(noise, 1-np.array(R2_noise),'k-o',markersize=2)
    ax1.loglog(noise, 1-np.array(R2),'k--',markersize=2)
    plt.xlabel(r"noise scalor $\eta$", fontsize=10)
    plt.ylabel(r"$1-R^2$", color='k', fontsize=10)

    ax2 = ax1.twinx()
    ax2.loglog(noise, np.array(MSE_noise), 'b-o',markersize=2)
    ax2.loglog(noise, np.array(MSE), 'b--',markersize=2)
    plt.ylabel(r"MSE", color='b', fontsize=10)
    plt.subplots_adjust(left=0.2,bottom=0.2,right=0.9)

    ax1.set_ylim([0.95*min(min(MSE_noise), min(R2_noise)), 1.05*(max(max(MSE_noise), max(R2_noise)))])
    ax2.set_ylim([0.95*min(min(MSE_noise), min(R2_noise)), 1.05*(max(max(MSE_noise), max(R2_noise)))])
    ax2.get_yaxis().set_ticks([])
    
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'R2MSE_OLS_noise.png'), transparent=True, bbox_inches='tight')
    return plt.show()


plot_franke_noise(method = 'ols')

        
        
    
