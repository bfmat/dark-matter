#!/usr/bin/env python3
"""Given output from the configuration average calculator, print the overall accuracy found in a region of a grid search"""
# Created by Brendon Matusch, November 2018

import sys

# Read the data from standard input, and take only the accuracy values
accuracy_values = [float(line.split()[3]) for line in sys.stdin.readlines()]
# Print the average accuracy over all configurations
print('Overall average accuracy:', sum(accuracy_values) / len(accuracy_values))
