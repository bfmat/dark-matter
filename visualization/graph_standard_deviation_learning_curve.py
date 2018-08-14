#!/usr/bin/env python3
"""Given a series of mean class-wise standard deviations for saved validation sets, graph a learning curve that shows the standard deviation as the system trains"""
# Created by Brendon Matusch, August 2018

import sys

import matplotlib.pyplot as plt

# Load all of the lines of text from standard input, and strip whitespace
lines = [line.strip() for line in sys.stdin.readlines()]
# Take only the lines that reference the standard deviation for a specific file (as opposed to an overall mean)
lines = [line for line in lines if line.startswith('Mean standard deviation for file')]
# Get standard deviation values by taking the last word in each line
standard_deviations = [float(line.split()[-1]) for line in lines]
# Plot the standard deviations on the Y axis
plt.plot(standard_deviations)
# Set the X and Y axis labels
plt.xlabel('Epoch')
plt.ylabel('Mean Class-Wise Standard Deviation')
# Display the graph on screen
plt.show()
