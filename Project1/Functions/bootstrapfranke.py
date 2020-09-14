import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
from numba import jit, njit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

# Add the src/ directory to the python path so we can import the code 
# we need to test directly as 'from <file name> import <function/class>'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Functions'))
from LinearReg import linregOwn, linregSKL
from Franke import franke
from designMat import designMatrix

linreg = linregOwn(method="ols")

##Generate franke values
N = int(1e4)
x1 = np.random.rand(N)
x2 = np.random.rand(N)
y = franke(x1, x2)

# Hold out some test data that is never used in training.
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, 
                                                                         test_size=0.2, random_state = 42)

X_test      = np.zeros(shape=(x1_test.shape[0],2))
X_test[:,0] = x1_test
X_test[:,1] = x2_test



