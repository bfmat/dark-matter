#!/usr/bin/env python3
"""A tool for graphing a validation data set, printing out standard deviation data, and comparing a neural network's output with the acoustic parameter or the original neural network score"""
# Created by Brendon Matusch, June 2018

import sys

import matplotlib.pyplot as plt
import numpy as np

from data_processing.experiment_serialization import load_test
from utilities.verify_arguments import verify_arguments

# Verify that a path to the JSON data file is passed
verify_arguments('JSON data file')

# Load the data set from the file
event_data_set, ground_truths, network_outputs = load_test(sys.argv[1])
# Get the acoustic parameter and neural network score data from the events
acoustic_parameters, original_neural_network_scores = zip(
    *((event.logarithmic_acoustic_parameter, event.original_neural_network_score)
      for event in event_data_set.validation_events)
)

# Iterate over the three criteria standard deviations will be calculated for, and corresponding names
for criterion_data, criterion_name in zip(
    [network_outputs, original_neural_network_scores],
    ['network outputs', 'original neural network scores']
):
    # Iterate over both possible values of the ground truth, and corresponding names
    for ground_truth_value, ground_truth_name in zip([True, False], ['alpha particles', 'neutrons']):
        # Sort out only the data points for this criterion which correspond to ground truths of this value
        data = np.array([
            criterion_data[index]
            for index in range(len(criterion_data))
            if ground_truths[index] == ground_truth_value
        ])
        # Subtract the ideal output for this ground truth value from each of the data points
        errors = data - int(ground_truth_value)
        # Calculate the geometric mean of these errors, which is the standard deviation off the ideal output
        squared_errors = errors ** 2
        variance = np.mean(squared_errors)
        standard_deviation = np.sqrt(variance)
        # Output the standard deviation to the user
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
    x=original_neural_network_scores,
    y=network_outputs,
    c=point_colors
)
# Label the X and Y axes
plt.xlabel('Original Neural Network Score')
plt.ylabel('Neural Network Prediction')
# Display the graph on screen
plt.show()
