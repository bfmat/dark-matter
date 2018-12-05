#!/usr/bin/env python3
"""Convert results from a pulse count grid search to a CSV format that can be put in a table"""
# Created by Brendon Matusch, December 2018

import csv
import sys

import numpy as np

# Read the log file from standard input, only taking the lines that include removal statistics
lines = [line for line in sys.stdin.readlines() if 'removed' in line]
# Read the numerical values into a NumPy array
data = np.array([float(line.split()[-1]) for line in lines])
# Reshape the array so the two values in the third dimension represents proportions of neck alphas and nuclear recoils removed, respectively
data = np.reshape(data, (len(data) // 2, 2))
# Convert it to a list so other data types can be added
data = data.tolist()

# Create a list of tuples sof all the combinations in order
hyperparameter_combinations = []
for l2_lambda in [0, 0.0003, 0.0006, 0.001, 0.003]:
    for dropout in [0, 0.25, 0.5]:
        for hidden_layers in [2, 4, 6]:
            for activation in ['relu', 'tanh']:
                hyperparameter_combinations.append([l2_lambda, dropout, hidden_layers, activation])

# print(len(data))
# print(len(hyperparameter_combinations))

# Create a CSV writer for standard output
writer = csv.writer(sys.stdout)
# Write the column titles
writer.writerow(['L2 Lambda', 'Dropout', 'Hidden Layers', 'Activation', 'Neck Alphas Removed', 'Simulated WIMPs Removed'])
# Iterate over hyperparameters with corresponding removal statistics
for removal_stats, hyperparameters in zip(data, hyperparameter_combinations):
    # Append them together into one list, which will be a CSV line
    data_line = hyperparameters + removal_stats
    # Write the line to CSV
    writer.writerow(data_line)
