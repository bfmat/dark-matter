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
    # Try to get the training and validation accuracy values out of lines containing them, skipping the last line because we need to add one to the index
    training_accuracy = []
    validation_accuracy = []
    for line_index, line in enumerate(test_lines[:-1]):
        # Lines at the end of the epoch include the word 'step', and the line after them is either a validation loss identifier or a path to a saved validation
        line_after = test_lines[line_index + 1]
        if 'step' in line and ('Validation' in line_after or 'Data saved at' in line_after):
            # Split the line by whitespace
            words = line.split()
            # Get the index of the accuracy identifier
            accuracy_identifier_index = words.index('acc:')
            # The accuracy value is at the next index; convert it to a number and add it to the list
            training_accuracy.append(float(words[accuracy_identifier_index + 1]))
            # Check if the validation accuracy identifier is in the line
            if 'val_acc:' in words:
                # If so, get the number out of the next index and add it to the list in the same way
                validation_accuracy_identifier_index = words.index('val_acc:')
                validation_accuracy.append(float(words[validation_accuracy_identifier_index + 1]))
    # If there were no validation accuracy values in the same lines, get them out of their own specific lines
    if not validation_accuracy:
        # It will be at the end of the lines
        validation_accuracy = [float(line.split()[2]) for line in test_lines if 'Validation accuracy:' in line]
    # Print the mean and maximum training and validation accuracy values
    print('Mean training accuracy:', sum(training_accuracy) / len(training_accuracy))
    print('Maximum training accuracy:', max(training_accuracy))
    print('Mean validation accuracy:', sum(validation_accuracy) / len(validation_accuracy))
    print('Maximum validation accuracy:', max(validation_accuracy))
    # Get the line index corresponding to the maximum validation accuracy
    # Start by finding all the paths to saved validation sets (which are at the end of their corresponding lines)
    validation_set_paths = [line.split()[3] for line in test_lines if 'Data saved at' in line]
    # Get the index of the maximum validation accuracy; the corresponding validation set will have the same index
    maximum_validation_accuracy_index = validation_accuracy.index(max(validation_accuracy))
    print('Maximum accuracy validation set is saved at', validation_set_paths[maximum_validation_accuracy_index])
