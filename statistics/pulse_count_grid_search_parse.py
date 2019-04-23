#!/usr/bin/env python3
"""A script to average the numbers of removed neck alphas and nuclear recoils over multiple configuration tests in a pulse count grid search"""
# Created by Brendon Matusch, November 2018

import sys

import numpy as np

# Recalculate the list of hyperparameters to add to the results sheet
hyperparameters = []
for l2_lambda in [0, 0.0003]:
    for activation in ['tanh']:
        for convolutional_layers in [2, 3]:
            for filters in [8, 16]:
                for dense_layers in [1, 2]:
                    for zero_weight in [0.005, 0.01, 0.015, 0.02]:
                        hyperparameters.append([l2_lambda, activation, convolutional_layers, filters, dense_layers, zero_weight])


# Read the log file from standard input, only taking the lines that include removal statistics
lines = [line for line in sys.stdin.readlines() if 'removed' in line]
# If there are 1564 such lines (this is the incomplete first half of the map projection search) cut it to its proper length
if len(lines) == 1564:
    lines = lines[:1536]
# Read the numerical values into a NumPy array
data = np.array([float(line.split()[-1]) for line in lines])
# Reshape the array so the two values in the third dimension represents proportions of neck alphas and nuclear recoils removed, respectively, and runs are separated into groups of six that are part of the same configuration
data = np.reshape(data, (len(data) // 24, 12, 2))
# Calculate the mean, max, min, and standard deviation for each configuration, concatenating them together
statistics = np.concatenate([
    np.mean(data, axis=1),
    np.std(data, axis=1),
    np.max(data, axis=1),
    np.min(data, axis=1)
], axis=1)
# mean_alpha, mean_wimp, std_alpha, std_wimp, max_alpha, max_wimp, min_alpha, min_wimp = statistics[configuration_index]
# Print an initial line for the CSV
print('L2 lambda,Activation,Conv layers,Filters,Dense layers,Alpha class weight,Mean alpha removal,Mean WIMP removal,Alpha removal std dev,WIMP removal std dev,Max alpha removal,Max WIMP removal,Min alpha removal,Min WIMP removal')
# Print out each of the good runs in a nicely formatted list
for configuration_index in range(len(statistics)):
    # Combine the hyperparameter and result table rows
    data_out = np.concatenate([hyperparameters[configuration_index], statistics[configuration_index]])
    # Output the combined row of data in CSV format
    print(*data_out, sep=',')
