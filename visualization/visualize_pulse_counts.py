#!/usr/bin/env python3
"""A tool for visualizing the number of pulses that each PMT receives in 3D space"""
# Created by Brendon Matusch, September 2018

import random

import matplotlib.pyplot as plt
from matplotlib import cm
# This import has to be done to register the 3D projection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from data_processing.load_deap_data import load_real_world_deap_data, load_simulated_deap_data
from data_processing.pmt_positions import X_POSITIONS, Y_POSITIONS, Z_POSITIONS

# Load all of the simulated data for plotting
neck_events, non_neck_events = load_simulated_deap_data()
# Choose a single neck event and a single non-neck event
neck_event, non_neck_event = (random.choice(event_list) for event_list in [neck_events, non_neck_events])
# Get the lists of pulse counts from each
neck_pulses, non_neck_pulses = (event[0] for event in [neck_event, non_neck_event])
# Get the overall maximum number of pulses so we can define the color spectrum
max_pulse_count = np.amax(np.concatenate([neck_pulses, non_neck_pulses]))
print(max_pulse_count)
# Divide each of the pulse counts by the maximum to get numbers from 0 to 1 (representing positions on the rainbow for plotting purposes)
neck_rainbow, non_neck_rainbow = (pulses.astype(float) / max_pulse_count for pulses in [neck_pulses, non_neck_pulses])
# Convert those positions on the rainbow to RGB colors
neck_colors, non_neck_colors = (cm.rainbow(rainbow_positions) for rainbow_positions in [neck_rainbow, non_neck_rainbow])

# Iterate over the lists of colors with corresponding plot indices
for colors, plot_index in zip([neck_colors, non_neck_colors], [1, 2]):
    # Create a subplot with 3D axes (stacking the subplots vertically)
    axes = plt.subplot(2, 1, plot_index, projection='3d')
    # Scatter plot the colors at the constant positions
    axes.scatter(X_POSITIONS, Y_POSITIONS, Z_POSITIONS, c=neck_colors)
# Display the 2 plots on screen
plt.show()
