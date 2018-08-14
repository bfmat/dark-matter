#!/usr/bin/env python3
"""Given a number of saved validation sets, calculate the class-wise standard deviation for each and print out the minimum"""
# Created by Brendon Matusch, August 2018

import glob
import os
import sys

import numpy as np

from data_processing.experiment_serialization import load_test
from utilities.verify_arguments import verify_arguments

# An expandable list of files using a wildcard should be provided
verify_arguments('saved validation sets using wildcard')
# Create a list for mean class-wise standard deviations
mean_standard_deviations = []
# Iterate over the files corresponding to the full path
file_paths = glob.glob(os.path.expanduser(sys.argv[1]))
for file_path in file_paths:
    # Load the data set from the file
    events, ground_truths, network_outputs = load_test(file_path)
    # If there are any absolutely equal network outputs, add NaN to the list and skip this example because it will possibly mess with the standard deviations (producing zero values)
    if len(network_outputs) != len(np.unique(network_outputs)):
        mean_standard_deviations.append(np.nan)
        continue
    # Get AP values from the events, and substitute those for the network outputs (temporary hack)
    network_outputs = [event.logarithmic_acoustic_parameter for event in events]
    # Divide all of the network outputs by their overall standard deviation, to normalize their range
    overall_standard_deviation = np.std(network_outputs)
    # Otherwise, go ahead and divide it in place
    network_outputs /= overall_standard_deviation
    # Create a list to add the standard deviations for each class to
    standard_deviations_by_class = []
    # Iterate over both possible values of the ground truth
    for ground_truth in [True, False]:
        # Sort out only the network outputs which correspond to ground truths of this value
        data = np.array([
            network_outputs[index]
            for index in range(len(network_outputs))
            if ground_truths[index] == ground_truth
        ])
        # Calculate the standard deviation of these values and append it to the list
        standard_deviations_by_class.append(np.std(data))
    # Calculate the mean of the 2 class-wise standard deviations to the list
    mean_standard_deviations.append(np.mean(standard_deviations_by_class))
# Print the lowest standard deviation (ignoring NaNs) and the corresponding file path
minimum_standard_deviation = np.nanmin(mean_standard_deviations)
print(f'Lowest standard deviation is {minimum_standard_deviation}, found in file {file_paths[mean_standard_deviations.index(minimum_standard_deviation)]}')
# Print the mean standard deviation over all files
print(f'Mean standard deviation is {np.nanmean(mean_standard_deviations)}')
