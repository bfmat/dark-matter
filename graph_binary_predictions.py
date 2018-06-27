#!/usr/bin/env python3
"""A tool for graphing a validation data set, and comparing a neural network's output with the acoustic parameter"""
# Created by Brendon Matusch, June 2018

import sys

from experiment_serialization import load_test
from verify_arguments import verify_arguments

# Verify that a path to the JSON data file is passed
verify_arguments('JSON data file')

# Load the data set from the file
event_data_set, ground_truths, network_outputs = load_test(sys.argv[1])
