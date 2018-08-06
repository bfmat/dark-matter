#!/usr/bin/env python3
"""A script for calculating the optimal value of each hyperparameter in a grid search"""
# Created by Brendon Matusch, August 2018

import os
import sys

import numpy as np

from utilities.verify_arguments import verify_arguments

# A path to an AP similarity log file is expected
verify_arguments('path to AP similarity log file')
# Load all lines in the file
with open(os.path.expanduser(sys.argv[1])) as file:
    lines = file.readlines()
# Take only the lines containing the mean disagreement statistics, and strip whitespace
mean_lines = [line.strip() for line in lines if 'Mean' in line]
# Extract the numeric disagreement values
mean_disagreements = [float(line.split()[1]) for line in mean_lines]
# Iterate over each of the hyperparameters that are used
for hyperparameter in ['dropout', 'l2_lambda', 'initial_examples', 'initial_threshold', 'threshold_multiplier']:
    # Create a dictionary to add lists of disagreement values according to hyperparameter values
    disagreement_by_hyperparameter_value = {}
    # Iterate over each of the mean lines with corresponding disagreement values
    for line, disagreements in zip(mean_lines, mean_disagreements):
        # Split the line by the name of the hyperparameter, and then get the hyperparameter value (as a string) up to the next underscore separator
        hyperparameter_value = line.split(hyperparameter)[1].split('_')[0]
        # If this value is not already in the dictionary, add a corresponding empty list
        if hyperparameter_value not in disagreement_by_hyperparameter_value:
            disagreement_by_hyperparameter_value[hyperparameter_value] = []
        # Add the corresponding disagreements to the list
        disagreement_by_hyperparameter_value[hyperparameter_value].append(disagreements)
    # Calculate the overall mean number of disagreements for each hyperparameter value, and get the corresponding values themselves
    overall_means, corresponding_values = zip(*((np.mean(disagreement_by_hyperparameter_value[hyper]), hyper) for hyper in disagreement_by_hyperparameter_value))
    # Find the index of the best number of disagreements, and print out the corresponding optimal hyperparameter value
    print(f'Optimal {hyperparameter}: {corresponding_values[np.argmin(overall_means)]}')
