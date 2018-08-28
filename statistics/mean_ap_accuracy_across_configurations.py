#!/usr/bin/env python3
"""Given standard input from the AP similarity statistic calculation script, compute the mean accuracy over each of the multiple runs for every configuration"""
# Created by Brendon Matusch, August 2018

import sys

import numpy as np

# Read all lines from standard input, and strip whitespace
input_lines = [line.strip() for line in sys.stdin.readlines()]
# Take only the lines that contain mean disagreement statistics
mean_lines = [line for line in input_lines if line.startswith('Mean')]
# Create a dictionary to hold the numbers of disagreements corresponding to each configuration (not including the run index)
disagreement_counts = {}
# Iterate over the mean lines, adding to the dictionary
for line in mean_lines:
    # Take the number of disagreements (for this specific run index) from the line
    disagreements = float(line.split()[1])
    # Take the full path from the line, removing the run index component to get the configuration
    configuration = line.split()[-1].split('configuration_test')[0]
    # If this configuration is not in the dictionary, create an empty list for it
    if configuration not in disagreement_counts:
        disagreement_counts[configuration] = []
    # Add the disagreement count to the corresponding list in the dictionary
    disagreement_counts[configuration].append(disagreements)
# Iterate over the configuration keys of the dictionary
for configuration in disagreement_counts:
    # Calculate the mean of the corresponding disagreement counts
    mean_disagreements = np.mean(disagreement_counts[configuration])
    # Convert that disagreement value to accuracy
    mean_accuracy = 1 - (mean_disagreements / 128)
    # Output the configuration name and accuracy to the user
    print(f'Mean accuracy for configuration {configuration} is {mean_accuracy}')
