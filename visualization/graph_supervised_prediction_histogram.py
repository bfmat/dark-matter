#!/usr/bin/env python3
"""Given a saved validation set from a supervised learning system, load it and display a histogram showing the distribution of the network's predictions for both ground truth classes"""
# Created by Brendon Matusch, August 2018

import sys

import matplotlib.pyplot as plt

from data_processing.experiment_serialization import load_test
from utilities.verify_arguments import verify_arguments

# Verify that a path to the JSON data file is passed
verify_arguments('JSON data file')

# Load the data set from the file
events, ground_truths, network_outputs = load_test(sys.argv[1])
# Separate the network's outputs based on the value of the corresponding ground truth
network_outputs_false = [output for output, ground_truth in zip(network_outputs, ground_truths) if not ground_truth]
network_outputs_true = [output for output, ground_truth in zip(network_outputs, ground_truths) if ground_truth]

# Set the size of the resulting graph (it should be standard across all such graphs)
plt.figure(figsize=(8, 6))
# Plot the network's outputs by ground truth in a histogram, labeling the 2 classes
plt.hist([network_outputs_false, network_outputs_true], bins=20, label=['Neutron Calibration Data', 'Alpha Background Data'])
# Label the axes of the graph
plt.xlabel('Network Prediction')
plt.ylabel('Count')
# Include a legend in the graph, explaining the colors
plt.legend()
# Display the graph on screen
plt.show()
