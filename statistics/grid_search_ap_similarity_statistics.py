#!/usr/bin/env python3
"""A script for searching a folder of saved validation sets, and finding the ones that behave the most like Acoustic Parameter"""
# Created by Brendon Matusch, August 2018

import glob
import os
import sys

import numpy as np

from data_processing.experiment_serialization import load_test
from utilities.verify_arguments import verify_arguments


# Create a dictionary to hold dictionaries of numbers of disagreements, arranged by a particular set of hyperparameters
disagreements_by_hyperparameters = {}


def load_disagreements(file_path: str) -> None:
    """Load a saved validation set and print the number of disagreements if it is below a certain threshold"""
    # Load the validation events and network outputs from the JSON file (ignoring the ground truths)
    events, _, network_outputs = load_test(file_path)
    # Convert the network outputs to binary predictions
    network_predictions = np.array([output >= 0.5 for output in network_outputs])
    # Do the same with the Acoustic Parameter
    ap_predictions = np.array([event.logarithmic_acoustic_parameter > 0.25 for event in events])
    # Calculate the number of events on which AP and the network disagree
    disagreements = np.count_nonzero(network_predictions != ap_predictions)
    # Get the full identifier of the run except for the timestamp
    run_identifier = file_path.split('time')[0]
    # If the identifier is not already in the dictionary, create a sub-dictionary
    if run_identifier not in disagreements_by_hyperparameters:
        disagreements_by_hyperparameters[run_identifier] = {}
    # In the sub-dictionary, add the number of disagreements, referenced by the specific path
    disagreements_by_hyperparameters[run_identifier][file_path] = disagreements


# An expandable list of files using a wildcard should be provided
verify_arguments('saved validation sets using wildcard')
# Get the files corresponding to the full path, allowing recursive searches
file_paths = glob.glob(os.path.expanduser(sys.argv[1]), recursive=True)
# Get the length of the list of file paths once, so it does not have to be calculated again
file_count = len(file_paths)
# Iterate over the files one by one with corresponding indices; these operations are not thread safe
for file_index, file_path in enumerate(file_paths):
    # Load and store information on the file
    load_disagreements(file_path)
    # Regularly print the number of files that have been loaded
    if file_index % 100 == 0:
        print(f'Loaded file {file_index} out of {file_count}')
# Iterate over the run identifiers in the dictionary
for run_identifier in disagreements_by_hyperparameters:
    # Print the mean number of disagreements throughout the entire run
    disagreement_values = list(disagreements_by_hyperparameters[run_identifier].values())
    print(f'Mean {np.mean(disagreement_values)} disagreements in run {run_identifier}')
    # Print the maximum number of disagreements throughout the run
    print(f'Maximum {np.amax(disagreement_values)} disagreements in run {run_identifier}')
    # Iterate over the specific file paths in this run
    for file_path in disagreements_by_hyperparameters[run_identifier]:
        # If there is a small number of disagreements, print out the path and number
        disagreements = disagreements_by_hyperparameters[run_identifier][file_path]
        if disagreements < 10:
            print(f'{disagreements} disagreements in file {file_path}')
    # Print a few blank lines for separation
    for _ in range(3):
        print()
