#!/usr/bin/env python3
"""A script to average the numbers of removed neck alphas and nuclear recoils over multiple configuration tests in a pulse count grid search"""
# Created by Brendon Matusch, November 2018

import sys

import numpy as np

# Recalculate the list of hyperparameters to add to the results sheet
hyperparameters = []
for num_convolutional_layers in [3, 6]:
    for l2_regularization in [0, 0.0005, 0.001, 0.003]:
        for nuclear_recoil_weight in [0.005, 0.01, 0.02]:
            for activation in ['relu', 'tanh']:
                hyperparameters.append([num_convolutional_layers, l2_regularization, nuclear_recoil_weight, activation])


# Read the log file from standard input, only taking the lines that include removal statistics
lines = [line for line in sys.stdin.readlines() if 'removed' in line]
# If there are 1564 such lines (this is the incomplete first half of the map projection search) cut it to its proper length
if len(lines) == 1564:
    lines = lines[:1536]
# Read the numerical values into a NumPy array
data = np.array([float(line.split()[-1]) for line in lines])
# Reshape the array so the two values in the third dimension represents proportions of neck alphas and nuclear recoils removed, respectively, and runs are separated into groups of six that are part of the same configuration
data = np.reshape(data, (len(data) // 12, 6, 2))
# Calculate the mean, max, min, and standard deviation for each configuration, concatenating them together
statistics = np.concatenate([
    np.mean(data, axis=1),
    np.std(data, axis=1),
    np.max(data, axis=1),
    np.min(data, axis=1)
], axis=1)
# Print an initial line for the CSV
print('Conv layers,L2 lambda,WIMP class weight,Activation,Mean alpha removal,Mean WIMP removal,Alpha removal std dev,WIMP removal std dev,Max alpha removal,Max WIMP removal,Min alpha removal,Min WIMP removal')
# Print out each of the good runs in a nicely formatted list
for configuration_index in range(len(statistics)):
    # Combine the hyperparameter and result table rows
    data_out = np.concatenate([hyperparameters[configuration_index], statistics[configuration_index]])
    # Output the combined row of data in CSV format
    print(*data_out, sep=',')
