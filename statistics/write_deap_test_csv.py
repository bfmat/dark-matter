#!/usr/bin/env python3
"""Write a saved DEAP JSON test set to easily parseable CSV data"""
# Created by Brendon Matusch, August 2018

import sys

from data_processing.deap_serialization import load_test
from utilities.verify_arguments import verify_arguments

# Verify that a path to the JSON data file is passed
verify_arguments('JSON data file')

# Load the data set from the file
ground_truths, network_outputs, identifiers = load_test(sys.argv[1])
# Print a line describing the fields of the CSV table
print('0AmBe1OneYear,0PredictedRecoil1PredictedNeckAlpha,RunID,SubRunID,EventID')
# Iterate over the ground truths, network outputs, and identifiers in sync
for ground_truth, network_output, identifier in zip(ground_truths, network_outputs, identifiers):
    # Print out both statistics and all 3 identifiers in one comma-separated line
    print(f'{ground_truth},{network_output},{identifier[0]},{identifier[1]},{identifier[2]}')
