#!/usr/bin/env python3
"""A script for searching a folder of saved validation sets, and finding the ones that behave the most like Acoustic Parameter"""
# Created by Brendon Matusch, August 2018

import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

from data_processing.experiment_serialization import load_test
from utilities.verify_arguments import verify_arguments


def load_and_print_disagreements(file_path: str) -> None:
    """Load a saved validation set and print the number of disagreements if it is below a certain threshold"""
    # Load the validation events and network outputs from the JSON file (ignoring the ground truths)
    events, _, network_outputs = load_test(file_path)
    # Convert the network outputs to binary predictions
    network_predictions = np.array([output >= 0.5 for output in network_outputs])
    # Do the same with the Acoustic Parameter
    ap_predictions = np.array([event.logarithmic_acoustic_parameter > 0.25 for event in events])
    # Calculate the number of events on which AP and the network disagree
    disagreements = np.count_nonzero(network_predictions != ap_predictions)
    # If the number of disagreements is below a certain threshold, print out the path and the number of disagreements
    if disagreements < 3:
        print(f'{file_path}: {disagreements} disagreements')


# An expandable list of files using a wildcard should be provided
verify_arguments('saved validation sets using wildcard')
# Get the files corresponding to the full path
file_paths = glob.glob(os.path.expanduser(sys.argv[1]))
# Load and print information on the files in parallel
with Pool(10) as thread_pool:
    thread_pool.map(load_and_print_disagreements, file_paths)
