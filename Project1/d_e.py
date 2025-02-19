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
from Crossvalidation import CrossValidation

#####Plot the MSE of the ridge and lasso fits for different values of lambda and polynomial degree


def MSE_Ridge_Lasso(method = 'lasso') :
    """
    Plots mean squared error (MSE) as a function of the polyomial degree (complexity parameter p) 
    for different values of the shrinkage parameter
    when lasso and ridge are used as the methods
    
    Arguments:
    method: character type, activated only when method = 'ridge' or 'lasso'
    """
    
    R2_noise           = []
    MSE_noise          = []
    MSE                = []
    R2                 = []

    k = 1
    fig, ax1 = plt.subplots()
    plt.rc('text', usetex=True)


    ind = -1

    for lambda_ in np.logspace(-4, 0, 5) :    ##10 is a default base 
        ind += 1
        MSE_noise = []

        for deg in range(1, 18) :
            if ind == 0 :
                linreg = linregOwn(method='ols')
            else :
                linreg = linregOwn(method=method)
            
            ##Compute MSE using cross-validation. Might take some time before we get plots, should be optimized with jit
            ##but no time to do that for all classes
            
            n = int(1000)
            np.random.seed(18271)
            x1 = np.random.rand(n)
            np.random.seed(91837)
            x2 = np.random.rand(n)
            y_data =  franke(x1, x2)           
            y_data_noise = y_data +  0.1 * np.random.standard_normal(size=n)
            CV_instance = CrossValidation(linreg, designMatrix)
            means_noise = CV_instance.kFoldCV(x1, x2, y_data_noise, 10, lambda_ = lambda_, degree = deg)
            means = CV_instance.kFoldCV(x1, x2, y_data, 10, lambda_ = lambda_, degree = deg)
            
            MSE_noise     .append(means_noise[0])
            MSE           .append(means[0])


        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        if ind == 0 :
            plt.plot(range(1, 18), np.array(MSE_noise), colors[ind]+'-o', markersize=5, label=r"OLS")
        else :
            plt.plot(range(1, 18), np.array(MSE_noise), colors[ind]+'-o', markersize=1, label=r"$\lambda=10^{%d}$"%(int(np.log10(lambda_))))
        
        plt.ylabel(r"MSE", fontsize=10)
        plt.xlabel(r"Polynomial degree $p$", fontsize=10)
        plt.subplots_adjust(left=0.2,bottom=0.2)

        #ax1.set_ylim([0.95*min(min(MSE_noise), min(R2_noise)), 1.05*(max(max(MSE_noise), max(R2_noise)))])
        
    ax1.legend()
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'MSE_lasso_noise_2.png'), transparent=True, bbox_inches='tight')
    plt.show()


MSE_Ridge_Lasso(method='ridge')
MSE_Ridge_Lasso(method='lasso')

###plot coefficients of beta

def plot_beta(method = 'ridge') :
    """
    Plots coefficients (beta) for ridge and lasso methods for different values of the shrinkage parameter (lambda)
    
    Arguments:
    method: character type, accepts arguments 'ridge' or 'lasso'
    """
    beta = []
    beta_variance = []

    k = 10000
    fig, ax1 = plt.subplots()
    plt.rc('text', usetex=True)


    ind = -1
    lam = np.logspace(-3, 5, 20)

    for lambda_ in lam :
        if ind == 0 :
            linreg = linregOwn(method='ols')
        else : 
            linreg = linregOwn(method=method)

        ind += 1
        n = int(500)
        np.random.seed(18271)
        x1 = np.random.rand(n)
        np.random.seed(91837)
        x2 = np.random.rand(n)
        y_data =  franke(x1, x2)
        eta = 0.1           
        y_data_noise = y_data +  eta * np.random.standard_normal(size=n)
        x1_train, x1_test, x2_train, x2_test, y_train, y_test, y_noise_train, y_noise_test = train_test_split(x1, x2, y_data,y_data_noise, test_size=0.35, random_state = 42)
        X_train = designMatrix(x1_train, x2_train, 3)
        X_test = designMatrix(x1_test, x2_test, 3)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_train[:, 0] = 1
        x_test = scaler.transform(X_test)
        X_test[:, 0] = 1
        linreg.fit(X_train, y_noise_train, lambda_)
        linreg.predict(X_test)
        var, low, up = linreg.CI(y_test)
        
        beta.append(linreg.fit(X_train, y_noise_train, lambda_)) ##Append lists together
        beta_variance.append(np.sqrt(var))

    beta = np.array(beta)
    print(beta)
    beta_variance = np.array(beta_variance)

    monomial = ['1',
                'x',
                'y',
                'x^2',
                'xy',
                'y^2',
                'x^3',
                'x^2y',
                'xy^2',
                'y^3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

    for i in range(10) :
        plt.errorbar(lam[1:], beta[1:,i], 
                        yerr=2*beta_variance[1:,i], 
                        fmt='-o',
                        markersize=2,
                        linewidth=1,
                        color=colors[i],
                        elinewidth=0.5,
                        capsize=2,
                        capthick=0.5,
                        label=r"$\beta_{%s}$"%(monomial[i]))
    plt.rc('text', usetex=True)
    plt.ylabel(r"$\beta_j$",  fontsize=10)
    plt.xlabel(r"shrinkage parameter $\lambda$",  fontsize=10)
    plt.subplots_adjust(left=0.2,bottom=0.2)
    plt.legend(fontsize=6)

    fig.gca().set_xscale('log')
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'beta_lasso.png'), transparent=True, bbox_inches='tight')
    plt.show()

plot_beta(method = 'ridge')
plot_beta(method = 'lasso')



#####Bias-variance tradeoff for different values of lambda

def plot_bias_var_tradeoff(ndegree = 5, method = 'ols'):
    """
    Plots bias-variance tradeoff using either bootstrap or cross-validation
    
    Arguments:
    ndegree: integer type, complexity of the model, number of polynomials
    sampling_method: character type, accepts arguments 'bootstrap' or 'cv'
    method: character type, accepts arguments 'ols', 'ridge' or 'lasso'
    """
    
    n = 500 ##make synthetic data
    np.random.seed(18271)
    x1 = np.random.rand(n)
    np.random.seed(91837)
    x2 = np.random.rand(n)
    y = franke(x1, x2) + 0.1*np.random.normal(0, 1, x1.size)
    bias = []
    var = []
    MSE = []
    
    for deg in range(1, ndegree + 1):
        bias_, var_, mse_, betavar_ = Bootstrap(x1, x2, y, degrees = deg, method = method)
        bias.append(bias_)
        var.append(var_)
        MSE.append(mse_)

    plot, ax = plt.subplots()
    plt.xlabel('Complexity (Order of polynomial)')
    plt.ylabel('MSE')
    
    if method == 'ols':
        plt.title('Bias-Variance tradeoff using bootstrap (OLS)')
    if method == 'ridge':
        plt.title('Bias-Variance tradeoff using bootstrap (Ridge)')
    if method == 'lasso':
        plt.title('Bias-Variance tradeoff using bootstrap (Lasso)')         
    
    plt.plot(range(1, ndegree+1), MSE, 'k-o', label = 'MSE') 
    plt.plot(range(1, ndegree+1), bias, 'b-o',label= 'Bias')
    plt.plot(range(1, ndegree+1), var, 'r-o',label = 'Variance') 
    #ax.axis([1,ndegree, 0, 1.1*np.max(var)]) 
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    plt.subplots_adjust(left=0.2,bottom=0.2,right=0.9)
    #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'lasso_bias_var.png'), transparent=True, bbox_inches='tight')

    return plt.show()



##MSE bias-variance tradeoff using bootstrap for different methods
plot_bias_var_tradeoff(ndegree=6, method = 'ols')

plot_bias_var_tradeoff(ndegree=6, method = 'ridge')

plot_bias_var_tradeoff(ndegree=6, method = 'lasso')