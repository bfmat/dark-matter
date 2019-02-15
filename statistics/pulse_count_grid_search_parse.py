#!/usr/bin/env python3
"""A script to average the numbers of removed neck alphas and nuclear recoils over multiple configuration tests in a pulse count grid search"""
# Created by Brendon Matusch, November 2018

import sys

import numpy as np

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
# Print out each of the good runs in a nicely formatted list
for configuration_index, (mean_alpha, mean_wimp, std_alpha, std_wimp, max_alpha, max_wimp, min_alpha, min_wimp) in enumerate(statistics):
    # # Print only configurations where the mean neck alpha removal is over 99.6% (in line with the conventional discriminator)
    # if mean_alpha < 0.996:
    #     continue
    # # Also filter for runs where less than 100% of WIMPs are removed (the network doesn't just output 1 for every example)
    # if mean_wimp == 1:
    #     continue
    # Now print out all of the relevant data
    print('CONFIGURATION', configuration_index)
    print('Mean alpha removal:', mean_alpha, 'Mean WIMP removal:', mean_wimp, 'Std dev of alpha removal:', std_alpha, 'Std dev of WIMP removal:', std_wimp)
    print('Max alpha removal:', max_alpha, 'Max WIMP removal:', max_wimp, 'Min alpha removal:', min_alpha, 'Min WIMP removal:', min_wimp)
    print('Neck alpha raw data:', sorted(data[configuration_index, :, 0].tolist()))
    print('WIMP raw data:', sorted(data[configuration_index, :, 1].tolist()))
    print()
