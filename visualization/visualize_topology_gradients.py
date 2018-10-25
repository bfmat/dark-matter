#!/usr/bin/env python3
"""A tool for visualizing the gradients of each PMT's input with respect to the output of the network"""
# Created by Brendon Matusch, September 2018

import json
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
# This import has to be done to register the 3D projection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from data_processing.pmt_positions import X_POSITIONS, Y_POSITIONS, Z_POSITIONS

# Load a list of gradients from input, parsing it as JSON
gradients = np.array(json.load(sys.stdin))
# Divide each of the gradients by the greatest absolute value to get numbers from 0 to 1
max_gradient = np.max(np.abs(gradients))
rainbow = gradients / max_gradient
# Convert those positions on the rainbow to RGB colors
neck_colors = cm.rainbow(rainbow)

# Create a subplot with 3D axes (stacking the subplots vertically)
axes = plt.subplot(projection='3d')
# Scatter plot the colors at the constant positions
axes.scatter(X_POSITIONS, Y_POSITIONS, Z_POSITIONS, c=neck_colors)
# Display the 2 plots on screen
plt.show()
