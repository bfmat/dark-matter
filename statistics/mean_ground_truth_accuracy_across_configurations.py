#!/usr/bin/env python3
"""Given standard input from the ground truth accuracy calculation script, aggregate the accuracy over the 3 different runs in each configuration"""

import sys

import numpy as np

# The number of lines printed by the original script on every run
LINES_PER_RUN = 12
# The number of runs with the same configuration, run in groups
RUNS_PER_CONFIGURATION = 3

# The indices corresponding to the lines containing (mean and maximum) (training and validation) validation accuracy values within a run
LINE_INDICES = range(7, 11)

# The index containing the path to the saved validation set (including the hyperparameters)
HYPERPARAMETER_LINE_INDEX = 11

# Read all lines from standard input
input_lines = sys.stdin.readlines()
# Calculate the total number of runs, based on the known number of lines per run
num_runs = len(input_lines) // LINES_PER_RUN
# Create lists to add the average mean training and validation accuracy values to
training_averages = []
validation_averages = []
# Iterate over each of the configurations, based on the number of runs and runs per configuration
for configuration_index in range(num_runs // RUNS_PER_CONFIGURATION):
    # Create (mean and maximum) (training and validation) accuracy accumulators for the configuration
    mean_training_accumulator = 0
    max_training_accumulator = 0
    mean_validation_accumulator = 0
    max_validation_accumulator = 0
    # Create a variable to store the hyperparameter path in
    hyperparameter_path = None
    # Store a corresponding value for the maximum validation accuracy
    max_validation_accuracy_for_hyperparameter_path = 0
    # Iterate over the runs within that configuration
    for run_index in range(RUNS_PER_CONFIGURATION):
        # Calculate the starting index of this run
        run_starting_index = ((configuration_index * RUNS_PER_CONFIGURATION) + run_index) * LINES_PER_RUN
        # Get the lines accuracy values, and split them accordingly
        mean_training, max_training, mean_validation, max_validation = [float(input_lines[run_starting_index + line_offset].split()[-1]) for line_offset in LINE_INDICES]
        # Add the accuracy values to the corresponding accumulators for the configuration
        mean_training_accumulator += mean_training
        max_training_accumulator += max_training
        mean_validation_accumulator += mean_validation
        max_validation_accumulator += max_validation
        # If the maximum validation accuracy on this run is better than the previous best
        if max_validation > max_validation_accuracy_for_hyperparameter_path:
            # Get the path containing the hyperparameters from the defined line and store it in the variable
            hyperparameter_path = input_lines[run_starting_index + HYPERPARAMETER_LINE_INDEX].split()[-1]
            # Also update the maximum validation accuracy, so if future runs are less accurate they will not overwrite the path
            max_validation_accuracy_for_hyperparameter_path = max_validation
    # Calculate the average accuracy statistics for this configuration, and output them to the user alongside the last hyperparameter path
    print(f'Average mean training accuracy for configuration {configuration_index} with hyperparameters {hyperparameter_path}: {mean_training_accumulator / RUNS_PER_CONFIGURATION}')
    print(f'Average maximum training accuracy for configuration {configuration_index} with hyperparameters {hyperparameter_path}: {max_training_accumulator / RUNS_PER_CONFIGURATION}')
    print(f'Average mean validation accuracy for configuration {configuration_index} with hyperparameters {hyperparameter_path}: {mean_validation_accumulator / RUNS_PER_CONFIGURATION}')
    print(f'Average maximum validation accuracy for configuration {configuration_index} with hyperparameters {hyperparameter_path}: {max_validation_accumulator / RUNS_PER_CONFIGURATION}')
    # Add the average mean values to their corresponding lists
    training_averages.append(mean_training_accumulator / RUNS_PER_CONFIGURATION)
    validation_averages.append(mean_validation_accumulator / RUNS_PER_CONFIGURATION)
# Calculate the Pearson correlation coefficient between the training and validation accuracy values
training_validation_correlation = np.corrcoef(training_averages, validation_averages)[0, 1]
# Output it to the user
print(f'Pearson correlation coefficient between average mean training and validation accuracy: {training_validation_correlation}')
