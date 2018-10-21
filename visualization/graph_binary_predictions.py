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

# Load the data set from the file, ignoring the run type ground truths
events, _, network_outputs = load_test(sys.argv[1])
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
# Calculate actual neutron/alpha ground truths based on AP
ground_truths = np.array(acoustic_parameters) > 0.25

# Iterate over the three criteria standard deviations will be calculated for, with corresponding names and classification thresholds
for criterion_data, criterion_name, classification_threshold in zip(
    copy.deepcopy([network_outputs, acoustic_parameters, original_neural_network_scores]),
    ['network outputs', 'acoustic parameters', 'original neural network scores'],
    [0.5, 0.25, 0.5]
):
    # Convert the raw outputs to binary classifications, based on whether they are greater than or equal to the threshold
    classifications = np.array([data_point >= classification_threshold for data_point in criterion_data])
    # Get the number of these classifications that agree with the ground truths
    agreements = np.count_nonzero(classifications == ground_truths)
    # Print out the resulting accuracy statistic to the user
    print(f'Accuracy of {agreements / 128} for {criterion_name}')
    # Repeat this process for precision (erroneous recoil predictions) and recall (erroneous alpha predictions) individually
    # First, count the number of events predicted as recoils, and then divide the number that are incorrectly predicted to be recoils by that
    predicted_recoils = np.count_nonzero(classifications == 0)
    erroneous_recoils = np.count_nonzero(np.logical_and(classifications == 0, ground_truths == 1))
    print(f'Precision of {(predicted_recoils - erroneous_recoils) / predicted_recoils} for {criterion_name}')
    # Repeat for the proportion of neutron calibration events that are correctly predicted as recoils; this represents not the purity, but the number missed
    actual_recoils = np.count_nonzero(ground_truths == 0)
    erroneous_alphas = np.count_nonzero(np.logical_and(classifications == 1, ground_truths == 0))
    print(f'Recall of {(actual_recoils - erroneous_alphas) / actual_recoils} for {criterion_name}')
    # Divide all of these data points by their overall standard deviation, to normalize their range
    criterion_data /= np.std(criterion_data)
    # Create a list for the 2 standard deviations
    standard_deviations = []
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
        # Add it to the list so the mean can be calculated
        standard_deviations.append(standard_deviation)
    # Calculate and print the mean class-wise standard deviation
    print(f'Mean class-wise standard deviation for {criterion_name} is {np.mean(standard_deviations)}')


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
background_patch = Patch(color='r', label='Alpha Particles')
calibration_patch = Patch(color='b', label='Neutrons')
# Display them in a legend
plt.legend(handles=[background_patch, calibration_patch])
# Label the X and Y axes
plt.xlabel('Logarithmic Acoustic Parameter')
plt.ylabel('Neural Network Prediction')
# Display the graph on screen
plt.show()
