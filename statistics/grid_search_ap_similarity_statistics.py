#!/usr/bin/env python3
"""A script for searching a folder of saved validation sets, and finding the ones that behave the most like Acoustic Parameter"""
# Created by Brendon Matusch, August 2018

import glob
import multiprocessing
import os
import sys
from typing import Tuple

import numpy as np

from data_processing.experiment_serialization import load_test
from utilities.verify_arguments import verify_arguments


def load_disagreements(file_path: str) -> Tuple[str, int, float, float, float, str]:
    """Load a saved validation set and return the run identifier, the number of disagreements, the precision and recall, the class-wise standard deviation, and the input file path"""
    # Load the validation events and network outputs from the JSON file (ignoring the ground truths)
    events, _, network_outputs = load_test(file_path)
    # Convert the network outputs to binary predictions
    network_predictions = np.array([output >= 0.5 for output in network_outputs])
    # Do the same with the Acoustic Parameter
    ap_predictions = np.array([event.logarithmic_acoustic_parameter > 0.25 for event in events])
    # Calculate the number of events on which AP and the network disagree
    disagreements = np.count_nonzero(network_predictions != ap_predictions)
    # Calculate precision (erroneous recoil predictions) and recall (erroneous alpha predictions) individually
    # First, count the number of events predicted as recoils, and then divide the number that are incorrectly predicted to be recoils by that
    predicted_recoils = np.count_nonzero(network_predictions == 0)
    erroneous_recoils = np.count_nonzero(np.logical_and(network_predictions == 0, ap_predictions == 1))
    precision = (predicted_recoils - erroneous_recoils) / predicted_recoils
    # Repeat for the proportion of neutron calibration events that are correctly predicted as recoils; this represents not the purity, but the number missed
    actual_recoils = np.count_nonzero(ap_predictions == 0)
    erroneous_alphas = np.count_nonzero(np.logical_and(network_predictions == 1, ap_predictions == 0))
    recall = (actual_recoils - erroneous_alphas) / actual_recoils
    # Get indices for the events that are considered neutrons and alphas by AP
    alpha_indices = np.where(ap_predictions)
    neutron_indices = np.where(np.logical_not(ap_predictions))
    # Divide the network's outputs by the standard deviation of the full set to normalize them to the same range as that of AP
    normalized_outputs = network_outputs / np.std(network_outputs)
    # Calculate the class-wise standard deviation, using the alphas and neutrons individually
    class_wise_standard_deviation = np.mean([np.std(normalized_outputs[indices]) for indices in [alpha_indices, neutron_indices]])
    # Get the full identifier of the run except for the timestamp
    run_identifier = file_path.split('time')[0]
    # Return the relevant outputs alongside the file path (passed directly through to make iteration easier)
    return run_identifier, disagreements, precision, recall, class_wise_standard_deviation, file_path


# An expandable list of files using a wildcard should be provided
verify_arguments('saved validation sets using wildcard')
# Get the files corresponding to the full path, allowing recursive searches
file_paths = glob.glob(os.path.expanduser(sys.argv[1]), recursive=True)
# Get the length of the list of file paths once, so it does not have to be calculated again
file_count = len(file_paths)
# Create a dictionary to hold dictionaries of numbers of disagreements, arranged by a particular set of hyperparameters
disagreements_by_hyperparameters = {}
# Get the run identifiers and corresponding numbers of disagreements using the file paths and corresponding indices, loading and processing files in parallel with a process pool
pool = multiprocessing.Pool(processes=10)
# Get a corresponding completion index for each file we iterate over
for file_index, (run_identifier, disagreements, precision, recall, class_wise_standard_deviation, file_path) in enumerate(pool.imap_unordered(load_disagreements, file_paths)):
    # If the run identifier is not already in the dictionary, create a sub-dictionary
    if run_identifier not in disagreements_by_hyperparameters:
        disagreements_by_hyperparameters[run_identifier] = {}
    # In the sub-dictionary, add the number of disagreements, the precision and recall, and the class-wise standard deviation, referenced by the specific path
    disagreements_by_hyperparameters[run_identifier][file_path] = (disagreements, precision, recall, class_wise_standard_deviation)
    # Regularly print the index of the latest file that has been loaded
    if file_index % 100 == 0:
        print(f'Loaded file {file_index} of {file_count}')

# Iterate over the run identifiers in the dictionary
for run_identifier in disagreements_by_hyperparameters:
    # Get lists of disagreements and class-wise standard deviations from the dictionary
    disagreement_values, precision_values, recall_values, class_wise_standard_deviations = zip(*disagreements_by_hyperparameters[run_identifier].values())
    # Print the run identifier and all 4 statistics in the same line
    print('Run:', run_identifier, 'CWSD:', np.mean(class_wise_standard_deviations), 'Disagreements:', np.mean(disagreement_values), 'Precision:', precision, 'Recall:', np.mean(recall_values))
    # Iterate over the specific file paths in this run
    for file_path in disagreements_by_hyperparameters[run_identifier]:
        # If there is a small number of disagreements, print out the path and number
        disagreements = disagreements_by_hyperparameters[run_identifier][file_path][0]
        if disagreements < 10:
            print(f'{disagreements} disagreements in file {file_path}')
    # Print a few blank lines for separation
    for _ in range(3):
        print()
