#!/usr/bin/env python3
"""A script for visualizing the optimal value of each hyperparameter in a semi-supervised learning grid search"""
# Created by Brendon Matusch, August 2018

import sys

import matplotlib.pyplot as plt
import numpy as np

# Read all lines provided to standard input (from the script that calculates mean by configuration) and strip whitespace
lines = [line.strip() for line in sys.stdin.readlines()]
# Get the configuration identifier from each line
configurations = [line.split()[1] for line in lines]
# Extract the accuracy value from each line
accuracy_values = [float(line.split()[3]) for line in lines]
# Create a dictionary to add accuracy dictionaries to, based on human-readable names
accuracy_dictionaries = {}
# Iterate over each of the hyperparameters that are used, alongside human-readable names
for hyperparameter, human_readable in [('dropout', 'Dropout'), ('l2_lambda', 'L2 Regularization Lambda'), ('initial_threshold', 'Initial Training Threshold'), ('threshold_multiplier', 'Training Threshold Multiplier'), ('initial_examples', 'Initial Examples')]:
    # Create a dictionary to add lists of accuracy values according to hyperparameter values
    accuracy_by_hyperparameter_value = {}
    # Iterate over each of the configurations with corresponding accuracy values
    for configuration, accuracy in zip(configurations, accuracy_values):
        # Split the line by the name of the hyperparameter, and then get the hyperparameter value (as a floating-point number) up to the next slash separator
        hyperparameter_value = float(configuration.split(hyperparameter)[1].split('/')[0])
        # If this value is not already in the dictionary, add a corresponding empty list
        if hyperparameter_value not in accuracy_by_hyperparameter_value:
            accuracy_by_hyperparameter_value[hyperparameter_value] = []
        # Add the corresponding accuracy to the list
        accuracy_by_hyperparameter_value[hyperparameter_value].append(accuracy)
    # Replace the lists of means by run with the overall mean for that hyperparameter value
    for hyperparameter_value in accuracy_by_hyperparameter_value:
        accuracy_by_hyperparameter_value[hyperparameter_value] = np.mean(accuracy_by_hyperparameter_value[hyperparameter_value])
    # Find the index of the best accuracy value, and print out the corresponding optimal hyperparameter value
    # The key and value views must be converted to lists, otherwise argmin will return an incorrect index
    print(f'Optimal {hyperparameter}: {list(accuracy_by_hyperparameter_value.keys())[np.argmax(list(accuracy_by_hyperparameter_value.values()))]}')
    # Add the dictionary to the dictionary for all hyperparameters
    accuracy_dictionaries[human_readable] = accuracy_by_hyperparameter_value

# Create a figure with a predefined size
plt.figure(figsize=(10, 12))
# Iterate over the human readable names of the 4 hyperparameters with a corresponding index for the plot
for plot_index, human_readable in enumerate(accuracy_dictionaries):
    # Select a subplot with the current index (plus 1, since Matplotlib is 1-indexed)
    plt.subplot(2, 3, plot_index + 1)
    # Get the hyperparameter values from the dictionary
    hyperparameter_values = list(accuracy_dictionaries[human_readable].keys())
    # Get the accuracy values corresponding to each of the values of this hyperparameter
    accuracy_values_for_hyperparameter = accuracy_dictionaries[human_readable].values()
    # Calculate the bar width, which should be 1/6 of the mean difference between a pair of examples
    bar_width = (max(hyperparameter_values) - min(hyperparameter_values)) / (6 * (len(hyperparameter_values) - 1))
    # Plot the accuracy values against the hyperparameter values in a bar graph
    plt.bar(hyperparameter_values, accuracy_values_for_hyperparameter, width=bar_width)
    # Set the title of the plot with the human-readable hyperparameter name
    plt.title(human_readable)
    # Set the X and Y axis titles
    plt.xlabel(human_readable)
    plt.ylabel('Mean Accuracy')
    # Limit the Y axis between 50% and 100%, to make the differences more apparent
    plt.ylim(0.5, 1)
# Display the graph on screen
plt.show()
