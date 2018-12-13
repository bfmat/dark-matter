#!/usr/bin/env python3
"""A script to average the numbers of removed neck alphas and nuclear recoils over multiple configuration tests in the topological CNN grid search"""
# Created by Brendon Matusch, November 2018

import sys

import numpy as np

# Read the log file from standard input, only taking the lines that include removal statistics
lines = [line for line in sys.stdin.readlines() if 'removed' in line][:1560]
# Read the numerical values into a NumPy array
data = np.array([float(line.split()[-1]) for line in lines])
# Reshape the array so the two values in the third dimension represents proportions of neck alphas and nuclear recoils removed, respectively, and runs are separated into groups of six that are part of the same configuration
data = np.reshape(data, (len(data) // 24, 12, 2))
# Calculate the mean, max, min, and standard deviation for each configuration, concatenating them together and outputting it as a nested list
data = np.concatenate([
    np.mean(data, axis=1),
    np.std(data, axis=1),
    np.max(data, axis=1),
    np.min(data, axis=1)
], axis=1)
a = data.tolist()
for b in a:
    if b[0] > 0.996:
        print(b)
