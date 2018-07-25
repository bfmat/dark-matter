#!/usr/bin/env python3
"""A script for analyzing and printing statistics about test runs in a grid search log"""
# Created by Brendon Matusch, July 2018

import os
import sys

from utilities.verify_arguments import verify_arguments

# Verify that a path to the log file is passed
verify_arguments('log file path')
# Get it from the command line arguments
log_path = os.path.expanduser(sys.argv[1])
# Read the full contents of the file into memory
with open(log_path) as log_file:
    log_data = log_file.read()
# Split the test into different tests based on the part where the hyperparameters are printed
# Remove data before the first test, which consists of system information printed at the beginning
tests = log_data.split('HYPERPARAMETERS')[1:]
# Iterate over the tests with corresponding indices, printing statistics for each one
for test_index, test in enumerate(tests):
    # Print a blank line for separation
    print()
    # Print out the index of this test
    print('Test', test_index)
    # Split the string into its component lines
    test_lines = test.split(os.linesep)
    # Strip whitespace from the lines, and remove all blank lines after this
    test_lines = [line.strip() for line in test_lines]
    test_lines = [line for line in test_lines if line]
    # Print out the hyperparameter description lines until the separator line that starts with an underscore
    hyperparameter_lines = []
    for line in test_lines:
        if line.startswith('_'):
            break
        print(line)
    # Get the training accuracy out of lines containing it; lines at the end of the epoch include the word 'step'
    training_accuracy = [float(line.split('acc:')[1].strip()) for line in test_lines if 'step' in line]
    # Get the validation accuracy out of lines containing it
    validation_accuracy = [float(line.split()[2]) for line in test_lines if 'Validation accuracy:' in line]
    # Print the mean and maximum training and validation accuracy values
    print('Mean training accuracy:', sum(training_accuracy) / len(training_accuracy))
    print('Maximum training accuracy:', max(training_accuracy))
    print('Mean validation accuracy:', sum(validation_accuracy) / len(validation_accuracy))
    print('Maximum validation accuracy:', max(validation_accuracy))
    # Get the line index corresponding to the maximum validation accuracy
    maximum_validation_line_index = test_lines.index(f'Validation accuracy: {max(validation_accuracy)}')
    # The next line contains the path to the corresponding validation data; get this and print it out
    validation_data_path = test_lines[maximum_validation_line_index + 1].split()[3]
    print('Maximum accuracy validation set is saved at', validation_data_path)
