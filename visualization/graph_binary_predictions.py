#!/usr/bin/env python3
"""A tool for graphing a validation data set, and comparing a neural network's output with the acoustic parameter"""
# Created by Brendon Matusch, June 2018

import sys

import matplotlib.pyplot as plt

from data_processing.experiment_serialization import load_test
from utilities.verify_arguments import verify_arguments

# Verify that a path to the JSON data file is passed
verify_arguments('JSON data file')

# Load the data set from the file
event_data_set, ground_truths, network_outputs = load_test(sys.argv[1])
# Get the acoustic parameter data from the events
acoustic_parameters = [
    event.logarithmic_acoustic_parameter
    for event in event_data_set.validation_events
]
# Convert the binary ground truth values to colors (red and blue) for graphing
point_colors = [
    'r' if ground_truth else 'b'
    for ground_truth in ground_truths
]
# Scatter plot the acoustic parameter on the X axis and the neural network's predictions on the Y axis
plt.scatter(
    x=acoustic_parameters,
    y=network_outputs,
    c=point_colors
)
# Label the X and Y axes
plt.xlabel('Logarithmic Acoustic Parameter')
plt.ylabel('Neural Network Prediction')
# Display the graph on screen
plt.show()
