#!/usr/bin/env python3
"""A tool for graphing a validation data set, printing out standard deviation data, and comparing a neural network's output with the acoustic parameter or the original neural network score"""
# Created by Brendon Matusch, June 2018

import copy
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from data_processing.experiment_serialization import load_test
from utilities.verify_arguments import verify_arguments

# Verify that a path to the JSON data file is passed
verify_arguments('JSON data file')

# Load the data set from the file
events, ground_truths, network_outputs = load_test(sys.argv[1])
# Get the acoustic parameter and neural network score data from the events, if they are present (if one is, both will be)
if hasattr(events[0], 'logarithmic_acoustic_parameter'):
    acoustic_parameters, original_neural_network_scores = zip(
        *((event.logarithmic_acoustic_parameter, event.original_neural_network_score)
          for event in events)
    )
# Otherwise, make them lists of random values of the same length as the network outputs
else:
    random_values = [random.uniform(0, 1) for _ in range(len(network_outputs))]
    acoustic_parameters = random_values
    original_neural_network_scores = random_values

    # Iterate over the three criteria standard deviations will be calculated for, and corresponding names
for criterion_data, criterion_name in zip(
    copy.deepcopy([network_outputs, acoustic_parameters,
                   original_neural_network_scores]),
    ['network outputs', 'acoustic parameters', 'original neural network scores']
):
    # Divide all of these data points by their overall standard deviation, to normalize their range
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
# Set the size of the resulting graph (it should be standard across all such graphs)
plt.figure(figsize=(8, 6))
# Scatter plot the acoustic parameter on the X axis and the neural network's predictions on the Y axis
plt.scatter(
    x=acoustic_parameters,
    y=network_outputs,
    c=point_colors
)
# Create patches that describe the 2 differently colored classes
background_patch = Patch(color='r', label='Background radiation runs')
calibration_patch = Patch(color='b', label='Neutron calibration runs')
# Display them in a legend
plt.legend(handles=[background_patch, calibration_patch])
# Label the X and Y axes
plt.xlabel('Logarithmic Acoustic Parameter')
plt.ylabel('Neural Network Prediction')
# Display the graph on screen
plt.show()
