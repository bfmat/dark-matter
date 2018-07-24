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
# Iterate over the tests, printing statistics for each one
for test in tests:
    # Split the string into its component lines
    test_lines = test.split(os.linesep)
    print(test_lines)
