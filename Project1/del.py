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

imageFile = os.path.join(os.path.dirname(__file__), 'Data', 'SRTM_data_Norway_' + str(1) + '.tif')
image = Image.open(imageFile, mode = 'r')
pixels = imread(imageFile)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(pixels)
plt.show()

print(pixels)
