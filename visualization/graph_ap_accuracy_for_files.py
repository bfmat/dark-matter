#!/usr/bin/env python3
"""Given a wildcard for a set of validation files, graph a learning curve"""
# Created by Brendon Matusch, March 2019

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from data_processing.experiment_serialization import load_test
from utilities.verify_arguments import verify_arguments

# An expandable list of files using a wildcard should be provided
verify_arguments('saved validation sets using wildcard', 'saved training sets using wildcard')
# Get the files corresponding to the full path, allowing recursive searches (ignoring folders)
file_paths = sorted(path for path in glob.glob(os.path.expanduser(sys.argv[1]), recursive=True) if os.path.isfile(path))

# Create a list of accuracy values by epoch
validation_accuracy_values = []
# Iterate over the files, extracting data
for file_path in file_paths:
    # Load the bubbles and outputs from the file
    bubbles, _, outputs = load_test(file_path)
    # Add up the number of validation outputs that match up with AP predictions
    correct = np.sum(np.array([bubble.logarithmic_acoustic_parameter > 0.25 for bubble in bubbles]) == np.rint(outputs))
    # Calculate the accuracy percentage based on this
    validation_accuracy_values.append(correct / len(outputs))

# REPEAT ALL THIS FOR TRAINING DATA
# Get the files corresponding to the full path, allowing recursive searches (ignoring folders)
file_paths = sorted(path for path in glob.glob(os.path.expanduser(sys.argv[2]), recursive=True) if os.path.isfile(path))
# Create a list of accuracy values by epoch
training_accuracy_values = []
# Iterate over the files, extracting data
for file_path in file_paths:
    # Load the bubbles and outputs from the file
    bubbles, _, outputs = load_test(file_path)
    # Add up the number of validation outputs that match up with AP predictions
    correct = np.sum(np.array([bubble.logarithmic_acoustic_parameter > 0.25 for bubble in bubbles]) == np.rint(outputs))
    # Calculate the accuracy percentage based on this
    training_accuracy_values.append(correct / len(outputs))

# Plot the training accuracy values in yellow and validation in green; label them accordingly
plt.plot(training_accuracy_values, 'C2', label='Training accuracy')
plt.plot(validation_accuracy_values, 'C4', label='Validation accuracy')
# Draw a legend with the labels that have already been set
plt.legend()
# Label the accuracy and epoch axes
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# Display the graph on screen
plt.show()
