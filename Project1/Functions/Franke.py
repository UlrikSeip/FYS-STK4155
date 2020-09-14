import numpy as np
from numba import jit

def franke(x, y):
    """ 
    Computes Franke function. 
    Franke's function has two Gaussian peaks of different heights, and a smaller dip. 
    It is used as a test function in interpolation problems.
    
    Franke's function is normally defined on the grid [0, 1] for each x, y.
    
    Arguments of the function:
    x : numpy array
    y : numpy array
    
    Output of the function:
    f : Franke function values at specific coordinate points of x and y
    """
    f = (0.75 * np.exp(-((9*x - 2)**2)/4  - ((9*y - 2)**2)/4 ) 
        + 0.75 * np.exp(-((9*x + 1)**2)/49 -  (9*y + 1)    /10) 
        + 0.5  * np.exp(-((9*x - 7)**2)/4  - ((9*y - 3)**2)/4 ) 
        - 0.2  * np.exp(-((9*x - 4)**2)    - ((9*y - 7)**2)   ))
    return f


