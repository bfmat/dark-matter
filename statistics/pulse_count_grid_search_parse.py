#!/usr/bin/env python3
"""A script to print out the results in the pulse count grid search (multiple runs are already averaged)"""
# Created by Brendon Matusch, November 2018

import sys

import numpy as np

# Read the log file from standard input, only taking the lines that include removal statistics
lines = [line for line in sys.stdin.readlines() if 'removed' in line]
# Read the numerical values into a NumPy array
data = np.array([float(line.split()[-1]) for line in lines])
# Reshape the array so the two values in the third dimension represents proportions of neck alphas and nuclear recoils removed, respectively
data = np.reshape(data, (len(data) // 2, 2))
# Output the resulting array as a nested list
print(data.tolist())
