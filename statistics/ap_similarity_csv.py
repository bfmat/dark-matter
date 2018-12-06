#!/usr/bin/env python3
"""Convert AP similarity statistics to a CSV spreadsheet"""
# Created by Brendon Matusch, December 2018

import csv
import re
import sys

import numpy as np

# Create a CSV writer to standard output
writer = csv.writer(sys.stdout)
# Print a line containing the hyperparameters and performance statistics
writer.writerow(['Dropout', 'L2 Lambda', 'Init Training Set', 'Init Threshold', 'Threshold Mult', 'Accuracy', 'Std Dev', 'Precision', 'Recall'])
# Load lines from standard input; only take those that contain a configuration
lines = [line for line in sys.stdin.readlines() if line.startswith('Configuration:')]
# Split each line by slashes; only take those segments that contain hyperparameters
lines_split = [line.split('/')[4:-1] for line in lines]
# Take numeric values from each of the values in the lines, taking the longest if there are multiple
hyper_numbers = [[max(re.findall(r'[-+]?\d*\.\d+|\d+', string), key=len) for string in line] for line in lines_split]
# Now take the performance statistics from the end of each line
performance_numbers = [[float(value) for value in np.array(line.split())[[3, 5, 7, 9]]] for line in lines]
# Iterate over the two lists together
for hyper, performance in zip(hyper_numbers, performance_numbers):
    # Concatenate the lists and output them to CSV
    writer.writerow(hyper + performance)
