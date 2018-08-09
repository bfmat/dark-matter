#!/usr/bin/env python3
"""A script for creating a 3-dimensional graph of the gravitational ground truth offset generation function, parameterized by the prediction and distortion power"""
# Created by Brendon Matusch, August 2018

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from training.gravitational_ground_truth_offsets import gravitational_ground_truth_offsets

# Create an array of possible predictions between 0 and 1 which will be graphed
possible_predictions = np.linspace(0, 1, 100)
# Create an array several distortion power options (not all of which are integers)
distortion_powers = np.linspace(3, 11, 100)
# Create mesh grids of all possible combinations
possible_predictions_mesh, distortion_powers_mesh = np.meshgrid(possible_predictions, distortion_powers)
# Vectorize the offset calculation function (inefficiently) so the entire grids can be processed in one call
offsets_vectorized = np.vectorize(gravitational_ground_truth_offsets)
# Run the function on the mesh grids, assuming a gravitational multiplier of 1
offsets_mesh = offsets_vectorized(possible_predictions_mesh, distortion_powers_mesh, 1)

# Create 3D axes in the default figure
axes = plt.gca(projection='3d')
# Plot the resulting on the vertical axis, with the predictions and distortion powers on the (X, Y) plane
axes.plot_surface(X=possible_predictions_mesh, Y=distortion_powers_mesh, Z=offsets_mesh)
# Label the 3 axes
axes.set_xlabel('Network Prediction $p$')
axes.set_ylabel('Distortion Power $\\psi$')
axes.set_zlabel('$GravDiff(p, \\psi, 1)$')
# Display the plot on screen
plt.show()
