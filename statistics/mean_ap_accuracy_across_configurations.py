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

# Iterate over the configuration keys of the dictionary
for configuration in statistics:
    # Zip the list of statistics into a NumPy array
    data = np.array(list(zip(*statistics[configuration])))
    # Convert the disagreement values to accuracy
    data[1] = 1 - (data[1] / VALIDATION_EXAMPLES)
    # Calculate the mean, standard deviation, minimum, and maximum of each statistic
    data = np.array([[statistic(row) for row in data] for statistic in [np.mean, np.std, np.min, np.max]])
    # Print the configuration followed by the data array
    print('Configuration:', configuration)
    print(data)
