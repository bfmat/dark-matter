#!/usr/bin/env python3
"""Visualize the mean accuracy values over the runs of a grid search"""
# Created by Brendon Matusch, August 2018

import os
import sys

import matplotlib.pyplot as plt

from utilities.verify_arguments import verify_arguments

# A path to an AP similarity log file is expected
verify_arguments('path to AP similarity log file')
# Load all lines in the file
with open(os.path.expanduser(sys.argv[1])) as file:
    lines = file.readlines()
# Take only the lines containing the mean disagreement statistics, and extract the numeric values
mean_disagreements = [float(line.split()[1]) for line in lines if 'Mean' in line]
# Calculate the corresponding accuracy values, by dividing by the total validation examples and subtracting from 1
accuracy_values = [1 - (disagreements / 128) for disagreements in mean_disagreements]

# Set the size of the resulting graph (it should be standard across all such graphs)
plt.figure(figsize=(8, 6))
# Plot the mean accuracy values by the run number in a bar graph
plt.bar(range(len(accuracy_values)), accuracy_values)
# Label the X and Y axes
plt.xlabel('Grid Search Configuration')
plt.ylabel('Mean Accuracy in Acoustic Parameter Replication')
# Display the graph on screen
plt.show()
