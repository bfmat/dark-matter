#!/usr/bin/env python3
"""Given standard input from the AP similarity statistic calculation script, compute the mean accuracy over each of the multiple runs for every configuration"""
# Created by Brendon Matusch, August 2018

import sys

import numpy as np

# The number of validation examples is different for the image grid search; add this special case as an argument
if len(sys.argv) > 1 and sys.argv[1] == 'img':
    VALIDATION_EXAMPLES = 316
else:
    VALIDATION_EXAMPLES = 128

# Read all lines from standard input, and strip whitespace
input_lines = [line.strip() for line in sys.stdin.readlines()]
# Take the lines that contain the run identifier with mean disagreement, standard deviation, precision, and recall statistics
statistic_lines = [line for line in input_lines if line.startswith('Run:')]
# Create a dictionary to hold the various statistics corresponding to each configuration (not including the run index)
statistics = {}
# Iterate over the statistic lines, adding to the dictionary
for line in statistic_lines:
    # Split the line into its component words
    words = line.split()
    # Take the full path from the line, removing the run index component to get the configuration
    configuration = words[1].split('configuration_test')[0]
    # Take all of the other statistics, converting them to numbers
    statistics_tuple = [float(words[word_index]) for word_index in [3, 5, 7, 9]]
    # If this configuration is not in the dictionary, create an empty list for it
    if configuration not in statistics:
        statistics[configuration] = []
    # Add the statistics tuple to the corresponding list in the dictionary
    statistics[configuration].append(statistics_tuple)

# Create a list of hyperparameters to output alongside the results
hyperparameters = []
for definitive_training_examples in [128, 256]:
    for gravity_multiplier_increment in [0.0005, 0.001, 0.003, 0.005, 0.008]:
        for learning_rate in [0.001, 0.003, 0.01, 0.03]:
            for distortion_power in [3, 5, 7, 9, 11]:
                hyperparameters.append([definitive_training_examples, gravity_multiplier_increment, learning_rate, distortion_power])
# Keep a list of the names of the hyperparameters, for searching of configurations
hyperparameter_names = ['definitive_training_examples', 'gravity_multiplier_increment', 'learning_rate', 'distortion_power']

# Output a starting line for the CSV
print('Labeled training examples,Gravitational multiplier increment,Learning rate,Distortion power,Mean squared error,Accuracy,Precision,Recall')
# Iterate over the hyperparameter combinations
for hyperparameter_list in hyperparameters:
    # Get the configuration corresponding to these hyperparameters
    right_configuration = None
    for configuration in statistics:
        # Iterate over each of the hyperparameter names, verifying that the right value is present
        right = True
        for index, hyperparameter_name in enumerate(hyperparameter_names):
            # Get the value after the name and before the underscore, and compare it to the current value being iterated over in the outer loop
            # If there is no underscore after the value, everything up until the end will be returned, so this is okay
            if configuration.split(hyperparameter_name)[1].split('/')[0] != str(hyperparameter_list[index]):
                right = False
                break
        # If we have the right one, set the variable outside the loop
        if right:
            right_configuration = configuration
            break
    # Zip the list of statistics into a NumPy array
    data = np.array(list(zip(*statistics[right_configuration])))
    # Convert the disagreement values to accuracy
    data[1] = 1 - (data[1] / VALIDATION_EXAMPLES)
    # Calculate the mean, standard deviation, minimum, and maximum of each statistic
    statistic_values = np.concatenate([[statistic(row) for row in data] for statistic in [np.nanmean]])
    # Concatenate together the hyperparameters and performance statistics
    out_values = np.concatenate([hyperparameter_list, statistic_values])
    # Print them out as a line of CSV data
    print(*out_values, sep=',')
