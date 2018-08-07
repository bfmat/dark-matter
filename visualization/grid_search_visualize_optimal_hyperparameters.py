#!/usr/bin/env python3
"""A script for calculating the optimal value of each hyperparameter in a grid search"""
# Created by Brendon Matusch, August 2018

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from utilities.verify_arguments import verify_arguments

# A path to an AP similarity log file is expected
verify_arguments('path to AP similarity log file')
# Load all lines in the file
with open(os.path.expanduser(sys.argv[1])) as file:
    lines = file.readlines()
# Remove lines that contain ground truth data from the gravitational differentiation grid search
lines = [line for line in lines if '_ground_truths' not in line]
# Take only the lines containing the mean disagreement statistics, and strip whitespace
mean_lines = [line.strip() for line in lines if 'Mean' in line]
# Extract the numeric disagreement values
mean_disagreements = [float(line.split()[1]) for line in mean_lines]
# Create a dictionary to add disagreement dictionaries to, based on human-readable names
disagreement_dictionaries = {}
# Iterate over each of the hyperparameters that are used, alongside human-readable names
# `distortion_root` is the old name for distortion power
for hyperparameter, human_readable in [('gravity_multiplier_increment', 'Gravity Multiplier Increment'), ('learning_rate', 'Learning Rate'), ('distortion_root', 'Distortion Power')]:
    # Create a dictionary to add lists of disagreement values according to hyperparameter values
    disagreement_by_hyperparameter_value = {}
    # Iterate over each of the mean lines with corresponding disagreement values
    for line, disagreements in zip(mean_lines, mean_disagreements):
        # Split the line by the name of the hyperparameter, and then get the hyperparameter value (as a floating-point number) up to the next underscore separator
        hyperparameter_value = float(line.split(hyperparameter)[1].split('_')[0])
        # If this value is not already in the dictionary, add a corresponding empty list
        if hyperparameter_value not in disagreement_by_hyperparameter_value:
            disagreement_by_hyperparameter_value[hyperparameter_value] = []
        # Add the corresponding disagreements to the list
        disagreement_by_hyperparameter_value[hyperparameter_value].append(disagreements)
    # Add the dictionary to the dictionary for all hyperparameters
    disagreement_dictionaries[human_readable] = disagreement_by_hyperparameter_value
    # Replace the lists of means by run with the overall mean for that hyperparameter value
    for hyperparameter_value in disagreement_by_hyperparameter_value:
        disagreement_by_hyperparameter_value[hyperparameter_value] = np.mean(disagreement_by_hyperparameter_value[hyperparameter_value])
    # Find the index of the best number of disagreements, and print out the corresponding optimal hyperparameter value
    # The key and value views must be converted to lists, otherwise argmin will return an incorrect index
    print(f'Optimal {hyperparameter}: {list(disagreement_by_hyperparameter_value.keys())[np.argmin(list(disagreement_by_hyperparameter_value.values()))]}')

# Create a figure with a predefined size
plt.figure(figsize=(10, 12))
# Iterate over the human readable names of the 4 hyperparameters with a corresponding index for the plot
for plot_index, human_readable in enumerate(disagreement_dictionaries):
    # Select a subplot with the current index (plus 1, since Matplotlib is 1-indexed)
    plt.subplot(2, 2, plot_index + 1)
    # Get the hyperparameter values from the dictionary
    hyperparameter_values = list(disagreement_dictionaries[human_readable].keys())
    # Calculate the corresponding accuracy values, by dividing by the total validation examples and subtracting from 1
    accuracy_values = [1 - (disagreements / 128) for disagreements in list(disagreement_dictionaries[human_readable].values())]
    # Calculate the bar width, which should be 1/6 of the mean difference between a pair of examples
    bar_width = (max(hyperparameter_values) - min(hyperparameter_values)) / (6 * (len(hyperparameter_values) - 1))
    # Plot the accuracy values against the hyperparameter values in a bar graph
    plt.bar(hyperparameter_values, accuracy_values, width=bar_width)
    # Set the title of the plot with the human-readable hyperparameter name
    plt.title(human_readable)
    # Set the X and Y axis titles
    plt.xlabel(human_readable)
    plt.ylabel('Mean Accuracy')
    # Limit the Y axis between 50% and 100%, to make the differences more apparent
    plt.ylim(0.5, 1)
# Display the graph on screen
plt.show()
