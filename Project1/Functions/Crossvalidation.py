##load libraries
import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
from numba import jit, njit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler


# Add the Functions/ directory to the python path so we can import the code 
sys.path.append(os.path.join(os.path.dirname(__file__), 'Functions'))

from LinearReg import linregOwn, linregSKL
from designMat import designMatrix
from Franke import franke

class CrossValidation:
    """  
        A class of cross-validation technique. Performs cross-validation with shuffling.
    """
    def __init__(self, LinearRegression, DesignMatrix):
        """
        Initialization
                
        Arguments:
        LinearRegression: Instance from the class created by either linregOwn or linregSKl
        DesignMatrix: Function that generates design matrix 
        """
        self.LinearRegression = LinearRegression
        self.DesignMatrix = DesignMatrix
    
    def kFoldCV(self, x1, x2, y, k = 10, lambda_ = 0, degree = 5):
        """
        Performs shuffling of the data, holds a split of the data as a test set at each split and evaluates the model
        on the rest of the data. 
        Calculates the MSE , R2_score, variance, bias on the test data and MSE on the train data.
        
        Arguments:
        x1: 1D numpy array
        x2: 1D numpy array
        y: 1D numpy array
        k: integer, the number of splits
        lambda_: float type, shrinkage parameter for ridge and lasso methods.
        degree: integer type, the number of polynomials, complexity parameter
        
        """
        self.lambda_ = lambda_
        M = x1.shape[0]//k   ## Split input data x in k folds of size M

        
        ##save the statistic in the list
        MSE_train = []
        MSE_k     = []
        R2_k      = []
        var_k     = []
        bias_k    = []
        
        ##shuffle the data randomly
        shf = np.random.permutation(x1.size)
        x1_shuff = x1[shf]
        x2_shuff = x2[shf]
        y_shuff = y[shf]
        
        for i in range(k):
            # x_k and y_k are the hold out data for fold k
            x1_k = x1_shuff[i*M:(i+1)*M]
            x2_k = x2_shuff[i*M:(i+1)*M]
            y_k = y_shuff[i*M:(i+1)*M]
            
            ## Generate train data and then scale both train and test
            index_true = np.array([True for i in range(x1.shape[0])])
            index_true[i*M:(i+1)*M] = False
            X_train = self.DesignMatrix(x1_shuff[index_true], x2_shuff[index_true], degree)
            y_train = y_shuff[index_true]
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_train[:, 0] = 1
 
            ### Fit the regression on the train data
            beta = self.LinearRegression.fit(X_train, y_train, lambda_)
            y_predict_train = np.dot(X_train, beta)
            MSE_train.append(np.sum( (y_train-y_predict_train)**2)/len(y_train))
            
            ## Predict on the hold out data and calculate statistic of interest
            X_k = self.DesignMatrix(x1_k, x2_k, degree)
            X_k = scaler.transform(X_k)
            X_k[:, 0] = 1
            y_predict = np.dot(X_k,beta)
            MSE_k.append(np.sum((y_k-y_predict)**2, axis=0, keepdims=True)/len(y_predict))
            R2_k.append(1.0 - np.sum((y_k - y_predict)**2, axis=0, keepdims=True) / np.sum((y_k - np.mean(y_k))**2, axis=0, keepdims=True) )
            var_k.append(np.var(y_predict,axis=0, keepdims=True))
            bias_k.append((y_k - np.mean(y_predict, axis=0, keepdims=True))**2 )        
        
        means = [np.mean(MSE_k), np.mean(R2_k), np.mean(var_k), 
                 np.mean(bias_k),np.mean(MSE_train)]
        #print('MSE_test: {}' .format(np.round(np.mean(MSE_k),3)))
        #print('R2: {}' .format(np.round(np.mean(R2_k),3)))
        #print('Variance of the predicted outcome: {}' .format(np.round(np.mean(var_k),3)))
        #print('Bias: {}' .format(np.round(np.mean(bias_k),3)))
        #print('MSE_train {}' .format(np.round(np.mean(MSE_train),3)))
        return means


