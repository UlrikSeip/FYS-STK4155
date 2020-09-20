import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from numba import jit
from sklearn.preprocessing import StandardScaler



# Add the Functions/ directory to the python path so we can import the code 
sys.path.append(os.path.join(os.path.dirname(__file__),  'Functions'))
from LinearReg import linregOwn, linregSKL
from Franke import franke
from designMat import designMatrix

def Bootstrap(x1,x2, y, N_boot=500, method = 'ols', degrees = 5, random_state = 42):
    """
    Computes bias^2, variance and the mean squared error using bootstrap resampling method
    for the provided data and the method.
    
    Arguments:
    x1: 1D numpy array, covariate
    x2: 1D numpy array, covariate
    N_boot: integer type, the number of bootstrap samples
    method: string type, accepts 'ols', 'ridge' or 'lasso' as arguments
    degree: integer type, polynomial degree for generating the design matrix
    random_state: integer, ensures the same split when using the train_test_split functionality
    
    Returns: Bias_vec, Var_vec, MSE_vec, betaVariance_vec
             numpy arrays. Bias, Variance, MSE and the variance of beta for the predicted model
    """
    ##split x1, x2 and y arrays as a train and test data and generate design matrix
    x1_train, x1_test,x2_train, x2_test, y_train, y_test = train_test_split(x1,x2, y, test_size=0.2, random_state = random_state)
    y_pred_test = np.zeros((y_test.shape[0], N_boot))
    X_test = designMatrix(x1_test, x2_test, degrees)
    
    betaMatrix = np.zeros((X_test.shape[1], N_boot))
    
    ##resample and fit the corresponding method on the train data
    for i in range(N_boot):
        x1_,x2_, y_ = resample(x1_train, x2_train, y_train)
        X_train = designMatrix(x1_, x2_, degrees)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_train[:, 0] = 1
        X_test = designMatrix(x1_test, x2_test, degrees)
        X_test = scaler.transform(X_test)
        X_test[:, 0] = 1
        
        if method == 'ols':
            manual_regression = linregOwn(method = 'ols')
            beta =  manual_regression.fit(X_train, y_)
        if method == 'ridge':
            manual_regression = linregOwn(method = 'ridge')
            beta =  manual_regression.fit(X_train, y_, lambda_ = 0.05)
        if method == 'lasso':
            manual_regression = linregOwn(method = 'lasso')
            beta =  manual_regression.fit(X_train, y_, lambda_ = 0.05)
            
        ##predict on the same test data
        y_pred_test[:, i] = np.dot(X_test, beta)
        betaMatrix[:, i] = beta
    y_test = y_test.reshape(len(y_test),1) 
      
    Bias_vec = []
    Var_vec  = []
    MSE_vec  = []
    betaVariance_vec = []
    R2_score = []
    y_test = y_test.reshape(len(y_test),1)
    MSE = np.mean( np.mean((y_test - y_pred_test)**2, axis=1, keepdims=True) )
    bias = np.mean( (y_test - np.mean(y_pred_test, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(y_pred_test, axis=1, keepdims=True) )
    betaVariance = np.var(betaMatrix, axis=1)
    print("-------------------------------------------------------------")
    print("Degree: %d" % degrees)
    print('MSE:', np.round(MSE, 3))
    print('Bias^2:', np.round(bias, 3))
    print('Var:', np.round(variance,3))
    print('{} >= {} + {} = {}'.format(MSE, bias, variance, bias+variance))
    print("-------------------------------------------------------------")
    
    Bias_vec.append(bias)
    Var_vec.append(variance)
    MSE_vec.append(MSE)
    betaVariance_vec.append(betaVariance)
    return Bias_vec, Var_vec, MSE_vec, betaVariance_vec
