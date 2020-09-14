### import libraries
import numpy as np
from numba import jit
from sklearn.preprocessing import StandardScaler

###Simple function
def designMatrix(x, y, k=5):
    """
    Generates the design matrix (covariates of polynomial degree k) and scales afterwards. 
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
    
    scaler = StandardScaler()
    xb = scaler.fit_transform(xb)
    xb[:, 0] = 1
    return xb

