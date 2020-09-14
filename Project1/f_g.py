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
from imageio import imread

# Add the src/ directory to the python path so we can import the code 
# we need to use directly as 'from <file name> import <function/class>'

sys.path.append(os.path.join(os.path.dirname(__file__), 'Functions'))
from LinearReg import linregOwn, linregSKL
from Franke import franke
from designMat import designMatrix
from Bootstrap import Bootstrap
from Crossvalidation import CrossValidation

def data(image_number = 1, plotting = True):
    """
    Extracts the terrain data
    
    Argumets:
    image_number: Integer type, accepting arguments either '1', '2'. Indicates the number of image
    plotting: Binary type, accepting arguments True/False. If True, 3D plot of the terrain data is produced
    
    """
    
    ##read the file located in the 'Data' folder
    imageFile = os.path.join(os.path.dirname(__file__), 'Data', 'SRTM_data_Norway_' + str(image_number) + '.tif')
    image = Image.open(imageFile, mode = 'r')
    image.mode = 'I' ##32-bit signed integer pixels
    x = np.linspace(0, 1, image.size[0])
    y = np.linspace(0, 1, image.size[1])
    X,Y = np.meshgrid(x,y)
    Z = np.array(image)
    ##do the min-max standardization
    Z = Z - np.min(Z)
    Z = Z / (np.max(Z)-np.min(Z))
    #print(X.shape)
    #print(Y.shape)
    #print(Z.shape)
    
    ##Extract every 60 row and every 30 cols
    XX = X[1::60, 1::30]
    YY = Y[1::60, 1::30]
    ZZ = Z[1::60, 1::30]
    #print(XX.shape)
    #print(YY.shape)
    #print(ZZ.shape)
    x1_train = XX.ravel()
    x2_train = YY.ravel()
    y_train = ZZ.ravel()
    
    ##Extract  every 60 row and every 30 cols, after 10th row and col, to produce
    ##a test data set different from the train data set
    XX = X[10::60, 10::30]
    YY = Y[10::60, 10::30]
    ZZ = Z[10::60, 10::30]
    x1_test = XX.ravel()
    x2_test = YY.ravel()
    y_test = ZZ.ravel()
    
    ##plot the data
    if plotting :
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(5))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.view_init(30, 150)
        #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'terrain.png'), transparent=True, bbox_inches='tight')
        plt.show()

    return x1_train, x2_train, y_train, x1_test, x2_test, y_test

data(plotting = True)

###Fit the model 
def fit_terrain(plotting=False) :
    """
    Fits OLS, Ridge and Lasso on the terrain data and plots the fit
    
    Arguments:
    plotting: Binary type, accepting arguments True/False. If True, plots the fit.
    """
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = data(image_number=2, plotting=False)

    for method in ['ols', 'ridge', 'lasso'] :
        lambda_ = 0.01
        linreg = linregOwn(method=method)
        
        X_train = designMatrix(x1_train, x2_train)
        print(X_train.shape)
        linreg.fit(X_train,y_train, lambda_ = lambda_)
        if method == 'ols':
            linreg.fit(X_train,y_train, lambda_ = 0)

        X_test = designMatrix(x1_test, x2_test)
        linreg.predict(X_test)
        print(linreg.MSE(y_test))

        if plotting :
            x = np.linspace(0, 1, 60)
            y = np.copy(x)
            XX,YY = np.meshgrid(x,y)
            print(XX.shape)
            ZZ = np.reshape(linreg.yHat, XX.shape)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(XX,YY,ZZ, cmap=cm.coolwarm,linewidth=0, antialiased=False)
            ax.set_zlim(-0.10, 1.40)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.view_init(30, 45+90)
            #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', method+'terrain.png'), transparent=True, bbox_inches='tight')
            plt.show()
            
    ##Plots the extracted test data
    if plotting :
        x = np.linspace(0, 1, 60)
        y = np.copy(x)
        XX,YY = np.meshgrid(x,y)
        ZZ = np.reshape(y_test, XX.shape)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(XX,YY,ZZ, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(5))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.view_init(30, 45+90)
        #plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'test_terrain.png'), transparent=True, bbox_inches='tight')
        plt.show()

fit_terrain(plotting = True)

##Plot MSE for each model by cross-validation
def MSE_terrain() :
    """
    Plots the mean squared error (MSE)  for OLS, Lasso and Ridge methods on the test data set for different values of
    the shrinkage parameter. Shrinkage parameter is ignored when OLS is used for fitting the data
    """
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = data(image_number=2, plotting=False)

    degree = 10
    ## Fit and predict ols
    linreg = linregOwn(method='ols')
    X_train = designMatrix(x1_train, x2_train, degree)
    linreg.fit(X_train,y_train)

    X_test = designMatrix(x1_test, x2_test, degree)
    linreg.predict(X_test)
    ols_MSE = linreg.MSE(y_test)
    ols_MSE = np.array([ols_MSE, ols_MSE])
    ##lambda just for plotting 
    ols_lambda = np.array([1e-5, 1])

    ###Choose lambda for ridge and fit and compute MSE on the test data
    ridge_lambda = np.logspace(-5,0,20)
    ridge_MSE = []
    for lambda_ in ridge_lambda : 
        print("ridge "+ str(lambda_))

        linreg = linregOwn(method='ridge')
        
        X_train = designMatrix(x1_train, x2_train, degree)
        linreg.fit(X_train,y_train, lambda_)

        X_test = designMatrix(x1_test, x2_test, degree)
        linreg.predict(X_test)
        ridge_MSE.append(linreg.MSE(y_test))
        print(linreg.MSE(y_test))

    ridge_MSE = np.array(ridge_MSE)
    
    ##Choose lambda for lasso and fit and compute MSE on the test data
    lasso_lambda = np.logspace(-4,0,20)
    lasso_MSE = []
    for lambda_ in lasso_lambda : 
        print("lasso " + str(lambda_))
        linreg = linregOwn(method='lasso')
        
        X_train = designMatrix(x1_train, x2_train, degree)
        linreg.fit(X_train,y_train, lambda_)

        X_test = designMatrix(x1_test, x2_test, degree)
        linreg.predict(X_test)
        lasso_MSE.append(linreg.MSE(y_test))

    lasso_MSE = np.array(ridge_MSE)
    
    ######################################################## plot
    plt.rc('text', usetex=True)

    plt.loglog(ols_lambda,   ols_MSE,   'k--o', markersize=1, linewidth=1, label=r'OLS')
    plt.loglog(ridge_lambda, ridge_MSE, 'r-o', markersize=1, linewidth=3, label=r'Ridge')
    plt.loglog(lasso_lambda, lasso_MSE, 'b-o', markersize=1, linewidth=1, label=r'Lasso')

    plt.xlabel(r"$\lambda$",  fontsize=10)
    plt.ylabel(r"MSE",  fontsize=10)
    plt.subplots_adjust(left=0.2,bottom=0.2)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'MSE_lambda_terrain.png'), transparent=True, bbox_inches='tight')
    plt.show()
    
MSE_terrain()