#!/usr/bin/env python3
"""Convert results from a pulse count grid search to a CSV format that can be put in a table"""
# Created by Brendon Matusch, December 2018

import csv
import json
import sys

import numpy as np

# Load performance data from standard input
data = json.load(sys.stdin)

# Create a list of tuples sof all the combinations in order
hyperparameter_combinations = []
for num_convolutional_layers in [3, 6]:
    for l2_regularization in [0, 0.0005, 0.001, 0.003]:
        for final_convolutional_layer_filters in [8]:
            for kernel_radius_2_first_layer in [0]:
                for nuclear_recoil_weight in [0.005, 0.01, 0.02]:
                    for activation in ['relu', 'tanh']:
                        hyperparameter_combinations.append([num_convolutional_layers, l2_regularization, final_convolutional_layer_filters,
                                                            kernel_radius_2_first_layer, nuclear_recoil_weight, activation])

# Create a CSV writer for standard output
writer = csv.writer(sys.stdout)
# Write the column titles
writer.writerow(['Conv Layers', 'L2 Lambda', 'Conv Filters', 'Kernel Radius', 'Class Weighting', 'Activation', 'Neck Alphas Removed', 'Simulated WIMPs Removed'])
# Iterate over hyperparameters with corresponding removal statistics
for removal_stats, hyperparameters in zip(data, hyperparameter_combinations):
    # Append them together into one list, which will be a CSV line
    data_line = hyperparameters + removal_stats
    # Write the line to CSV
    writer.writerow(data_line)
