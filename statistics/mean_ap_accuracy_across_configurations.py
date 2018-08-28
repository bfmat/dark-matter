#!/usr/bin/env python3
"""Given standard input from the AP similarity statistic calculation script, compute the mean accuracy over each of the multiple runs for every configuration"""
# Created by Brendon Matusch, August 2018

import sys

import numpy as np

# Read all lines from standard input, and strip whitespace
input_lines = [line.strip() for line in sys.stdin.readlines()]
# Take the lines that contain mean disagreement statistics
mean_disagreement_lines = [line for line in input_lines if line.startswith('Mean disagreements')]
# Create a dictionary to hold the numbers of disagreements corresponding to each configuration (not including the run index)
disagreement_counts = {}
# Iterate over the mean disagreement lines, adding to the dictionary
for line in mean_disagreement_lines:
    # Take the number of disagreements (for this specific run index) from the line
    disagreements = float(line.split()[3])
    # Take the full path from the line, removing the run index component to get the configuration
    configuration = line.split()[-1].split('configuration_test')[0]
    # If this configuration is not in the dictionary, create an empty list for it
    if configuration not in disagreement_counts:
        disagreement_counts[configuration] = []
    # Add the disagreement count to the corresponding list in the dictionary
    disagreement_counts[configuration].append(disagreements)

# Take the lines that contain mean class-wise standard deviation statistics
mean_standard_deviation_lines = [line for line in input_lines if line.startswith('Mean class-wise')]
# Create a dictionary to hold the standard deviations corresponding to each configuration (not including the run index)
standard_deviation_values = {}
# Iterate over the mean standard deviation lines, adding to the dictionary
for line in mean_standard_deviation_lines:
    # Take the standard deviation (for this specific run index) from the line
    standard_deviation = float(line.split()[5])
    # Take the full path from the line, removing the run index component to get the configuration
    configuration = line.split()[-1].split('configuration_test')[0]
    # If this configuration is not in the dictionary, create an empty list for it
    if configuration not in standard_deviation_values:
        standard_deviation_values[configuration] = []
    # Add the standard deviation to the corresponding list in the dictionary
    standard_deviation_values[configuration].append(standard_deviation)

# Iterate over the configuration keys of both dictionaries (the keys should be the same)
for configuration in disagreement_counts:
    # Calculate the mean of the corresponding disagreement counts
    mean_disagreements = np.mean(disagreement_counts[configuration])
    # Convert that disagreement value to accuracy
    mean_accuracy = 1 - (mean_disagreements / 128)
    # Output the configuration name and accuracy to the user
    print(f'Mean accuracy for configuration {configuration} is {mean_accuracy}')
    # Calculate and output the mean class-wise standard deviation to the user
    print(f'Mean class-wise standard deviation for configuration {configuration} is {np.mean(standard_deviation_values[configuration])}')
    # Print a blank line for separation
    print()
