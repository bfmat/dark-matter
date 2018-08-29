#!/usr/bin/env python3
"""Given standard input from the AP similarity statistic calculation script, compute the mean accuracy over each of the multiple runs for every configuration"""
# Created by Brendon Matusch, August 2018

import sys

import numpy as np

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
    # Zip the list of statistics and calculate the mean of each individual statistic
    mean_standard_deviation, mean_disagreements, mean_precision, mean_recall = [np.mean(statistic_list) for statistic_list in zip(*statistics[configuration])]
    # Convert the disagreement value to accuracy
    mean_accuracy = 1 - (mean_disagreements / 128)
    # Output the relevant statistics for this configuration to the user
    print('Configuration:', configuration, 'Accuracy:', mean_accuracy, 'CWSD:', mean_standard_deviation, 'Precision:', mean_precision, 'Recall:', mean_recall)
