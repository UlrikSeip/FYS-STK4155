### import libraries
import numpy as np
from numba import jit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os


# Add the Functions/ directory to the python path so we can import the code 
sys.path.append(os.path.join(os.path.dirname(__file__), 'Functions'))
from Franke import franke


###Simple function
def designMatrix(x, y, k=5):
    """
    Generates the design matrix (covariates of polynomial degree k). 
    Intercept is included in the design matrix. 
    Scaling does not apply to the intercept term.
    if k = 2, generated column vectors: 1, x, y, x^2, xy, y^2 
    if k = 3, generated column vectors: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    ...
    
    Arguments:
    x: 1D numpy array
    y: 1D numpy array
    k: integer type. complexity parameter (i.e polynomial degree) 
    """
    
    xb = np.ones((x.size, 1))
    
    for i in range(1, k+1):
        for j in range(i+1):
            xb = np.c_[xb, (x**(i-j))*(y**j)]

    xb[:, 0] = 1
    return xb
