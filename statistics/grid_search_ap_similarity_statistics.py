#!/usr/bin/env python3
"""A script for searching a folder of saved validation sets, and finding the ones that behave the most like Acoustic Parameter"""
# Created by Brendon Matusch, August 2018

import glob
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
    # Return the relevant outputs alongside the file path (passed directly through to make iteration easier)
    return disagreements, precision, recall, class_wise_standard_deviation, file_path


# An expandable list of files using a wildcard should be provided
verify_arguments('saved validation sets using wildcard')
# Get the files corresponding to the full path, allowing recursive searches (ignoring folders)
file_paths = [path for path in glob.glob(os.path.expanduser(sys.argv[1]), recursive=True) if os.path.isfile(path)]
# Filter the paths, taking only the ones that are in the last few epochs
epochs = [int(path.split('epoch')[1].split('.json')[0]) for path in file_paths]
max_epoch = max(epochs)
file_paths = [path for path, epoch in zip(file_paths, epochs) if epoch > max_epoch - 300]
# Get the run identifier corresponding to each of the file paths (not including the specific epoch)
run_identifiers = [file_path.split('time')[0] for file_path in file_paths]
# Convert the identifiers to a set so that they are all unique (there are many duplicated)
run_identifiers = set(run_identifiers)
# Iterate over each of the run identifiers, calculating the results for them
for run_identifier in run_identifiers:
    # Take only the file paths that are part of this run
    run_file_paths = [file_path for file_path in file_paths if file_path.startswith(run_identifier)]
    # Get the number of files in this run so we only have to calculate it once
    file_count = len(run_file_paths)
    # Create a dictionary to hold tuples of result values for the current run (standard deviation, accuracy, et cetera) indexed by file paths
    results = {}
    # Get the run identifiers and corresponding numbers of disagreements using the file paths and corresponding indices
    # Get a corresponding completion index for each file we iterate over
    for file_index, (disagreements, precision, recall, class_wise_standard_deviation, file_path) in enumerate([load_disagreements(path) for path in run_file_paths]):
        # In the dictionary, add the number of disagreements, the precision and recall, and the class-wise standard deviation, referenced by the specific path
        results[file_path] = (disagreements, precision, recall, class_wise_standard_deviation)
        # Regularly print the index of the latest file that has been loaded
        if file_index % 100 == 0:
            print(f'Loaded file {file_index} of {file_count}')
    # Get lists of individual results from the combined dictionary
    disagreement_values, precision_values, recall_values, class_wise_standard_deviations = zip(*results.values())
    # Print the run identifier and all 4 statistics in the same line
    print('Run:', run_identifier, 'CWSD:', np.mean(class_wise_standard_deviations), 'Disagreements:', np.mean(
        disagreement_values), 'Precision:', np.mean(precision_values), 'Recall:', np.mean(recall_values))
    # Iterate over the specific file paths in this run
    for file_path in results:
        # If there is a small number of disagreements, print out the path and number
        disagreements = results[file_path][0]
        if disagreements < 10:
            print(f'{disagreements} disagreements in file {file_path}')
    # Print a few blank lines for separation
    for _ in range(3):
        print()
