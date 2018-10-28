#!/usr/bin/env python3
"""Given mean accuracy values by configuration for a semi-supervised learning grid search, run a linear regression to predict accuracy based on the hyperparameter values and determine how well they are correlated"""
# Created by Brendon Matusch, Augsut 2018

import sys

import numpy as np
from sklearn.linear_model import LinearRegression

# The hyperparameters that we are concerned with using to predict accuracy
HYPERPARAMETERS = ['dropout', 'l2_lambda', 'initial_threshold', 'threshold_multiplier']

# Read all configuration lines provided to standard input (from the script that calculates mean by configuration) and strip whitespace
lines = [line.strip() for line in sys.stdin.readlines() if 'Configuration:' in line]
# Get the configuration identifier from each line
configurations = [line.split()[1] for line in lines]
# Create a NumPy array that will contain values of the hyperparameters
# It is okay if not all of the configurations are unique because of disregarded hyperparameters; they can be used for training individually
hyperparameter_values = np.empty((len(configurations), len(HYPERPARAMETERS)))
# Iterate over the configurations with corresponding indices, extracting arrays of hyperparameters
for configuration_index, configuration in enumerate(configurations):
    # Iterate over the hyperparameters with corresponding indices, extracting them one by one
    for hyperparameter_index, hyperparameter in enumerate(HYPERPARAMETERS):
        # Split the line by the name of the hyperparameter, and then get the hyperparameter value (as a floating-point number) up to the next slash separator
        hyperparameter_values[configuration_index, hyperparameter_index] = float(configuration.split(hyperparameter)[1].split('/')[0])
# Extract the accuracy value from each line
accuracy_values = [float(line.split()[3]) for line in lines]
# Create a linear regression classifier and train it to predict the accuracy values based on the hyperparameters including polynomials
linear_regression = LinearRegression()
linear_regression.fit(hyperparameter_values, accuracy_values)
# Print the classifier's score on its training set to get an idea of the correlation between the hyperparameters and accuracy
score = linear_regression.score(hyperparameter_values, accuracy_values)
print('Score:', score)
