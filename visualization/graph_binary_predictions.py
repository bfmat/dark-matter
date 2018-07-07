#!/usr/bin/env python3
"""A tool for graphing a validation data set, printing out standard deviation data, and comparing a neural network's output with the acoustic parameter or the original neural network score"""
# Created by Brendon Matusch, June 2018

import copy
import sys

import matplotlib.pyplot as plt
import numpy as np

from data_processing.experiment_serialization import load_test
from utilities.verify_arguments import verify_arguments

# Verify that a path to the JSON data file is passed
verify_arguments('JSON data file')

# Load the data set from the file
event_data_set, ground_truths, network_outputs = load_test(sys.argv[1])
# Temporary: shift the network's outputs down and run an offset sigmoid transform on them
network_outputs -= 0.9
network_outputs = np.tanh(network_outputs)
network_outputs += 1
network_outputs /= 2
# Get the acoustic parameter and neural network score data from the events
acoustic_parameters = [
    event.logarithmic_acoustic_parameter
    for event in event_data_set.validation_events
]

# Iterate over the three criteria standard deviations will be calculated for, and corresponding names
for criterion_data, criterion_name in zip(
    copy.deepcopy([network_outputs, acoustic_parameters]),
    ['network outputs', 'acoustic parameters']
):
    # Divide all of these data points by the overall standard deviation, to normalize their range
    criterion_data /= np.std(criterion_data)
    # Iterate over both possible values of the ground truth, and corresponding names
    for ground_truth_value, ground_truth_name in zip([True, False], ['alpha particles', 'neutrons']):
        # Sort out only the data points for this criterion which correspond to ground truths of this value
        data = np.array([
            criterion_data[index]
            for index in range(len(criterion_data))
            if ground_truths[index] == ground_truth_value
        ])
        # Calculate the standard deviation of these values and output it to the user
        standard_deviation = np.std(data)
        print(
            f'The standard deviation of {criterion_name}',
            f'for {ground_truth_name}',
            f'is {standard_deviation}'
        )

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
