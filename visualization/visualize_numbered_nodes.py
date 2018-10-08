#!/usr/bin/env python3
"""A tool for visualizing the DEAP detector topology, with each node numbered so that a graph can be constructed manually"""
# Created by Brendon Matusch, October 2018

import matplotlib.pyplot as plt
# This import has to be done to register the 3D projection
from mpl_toolkits.mplot3d import Axes3D

from data_processing.pmt_positions import X_POSITIONS, Y_POSITIONS, Z_POSITIONS

# Create a subplot with 3D axes (stacking the subplots vertically)
axes = plt.subplot(111, projection='3d')
# Iterate over the 3D points with corresponding numeric indices for labels
for point_index, point_3d in enumerate(zip(X_POSITIONS, Y_POSITIONS, Z_POSITIONS)):
    # If the point is above or below the center line (this will change), remove it so that it is easier to see the points on one side
    if point_3d[2] < -0.3 or point_3d[2] > 0.5:
        # Just skip this point, not plotting the point or the label
        continue
    # Scatter plot the point in the 3D graph
    axes.scatter(*point_3d)
    # Create a label with the corresponding index at that point
    axes.text(*point_3d, point_index)
# Display the plot on screen
plt.show()
