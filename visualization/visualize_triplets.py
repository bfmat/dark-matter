#!/usr/bin/env python3
"""A script for visualizing AP values and original neural network scores of triplet events compared to their defined classes"""
# Created by Brendon Matusch, August 2018

import matplotlib.pyplot as plt

from data_processing.load_triplet_classification_data import load_triplet_classification_data

# Load the triplet events
loud_events, quiet_events = load_triplet_classification_data()
# Get the AP values and neural network scores of the combined events
all_events = loud_events + quiet_events
ap_values = [event.logarithmic_acoustic_parameter for event in all_events]
neural_network_scores = [event.original_neural_network_score for event in all_events]
# Create a list of colors where the loud events are red and the quiet events are blue
colors = (['r'] * len(loud_events)) + (['b'] * len(quiet_events))
# Scatter plot the events with AP on the X axis, neural network predictions on the Y axis, and corresponding colors
plt.scatter(ap_values, neural_network_scores, c=colors)
# Label the X and Y axes
plt.xlabel('Logarithmic Acoustic Parameter')
plt.ylabel('Original Neural Network Score')
# Display the graph on screen
plt.show()
