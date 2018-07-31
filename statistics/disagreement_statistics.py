#!/usr/bin/env python3
"""A script to analyze a saved validation set, find the events on which the AP and neural network disagree, and print print information about those events"""
# Created by Brendon Matusch, July 2018

import sys

import numpy as np

from data_processing.experiment_serialization import load_test
from utilities.verify_arguments import verify_arguments

# Verify that a path to the JSON data file is passed
verify_arguments('JSON data file')
# Load the data set from the file
events, ground_truths, network_outputs = load_test(sys.argv[1])
# Get the acoustic parameter data from the events
acoustic_parameters = [event.logarithmic_acoustic_parameter for event in events]
# Convert the network's outputs to binary predictions (as a NumPy array)
network_predictions = np.array([output >= 0.5 for output in network_outputs])
# Repeat with acoustic parameters, dividing at a specific threshold
acoustic_predictions = np.array([acoustic_parameter >= 0.22 for acoustic_parameter in acoustic_parameters])
# Get the indices of the bubbles on which the network and the acoustic parameter disagree
disagreement_indices = np.argwhere(network_predictions != acoustic_predictions)
# Get the events on which the two techniques disagree (indices are wrapped in single-element arrays)
disagreement_events = [events[index[0]] for index in disagreement_indices]
# Iterate over the events, printing information
for event in disagreement_events:
    # Print several attributes of the event that make it identifiable
    print('Unique index:', event.unique_bubble_index)
    print('Date:', event.date)
    print('Run number:', event.run_number)
    print('Event number:', event.event_number)
    print('Run type:', event.run_type)
    # Print out the banded Fourier transform data originating in the ROOT file
    print('Banded frequency domain:', event.banded_frequency_domain)
    # Print a blank line for separation
    print()
