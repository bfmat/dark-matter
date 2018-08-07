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
# Get the original neural network scores from the events
original_neural_network_scores = [event.original_neural_network_score for event in events]
# Convert the network's outputs to binary predictions (as a NumPy array)
network_predictions = np.array([output >= 0.5 for output in network_outputs])
# Repeat with original neural network scores
original_predictions = np.array([score >= 0.5 for score in original_neural_network_scores])
# Get the indices of the bubbles on which the network and the acoustic parameter disagree
disagreement_indices = np.argwhere(network_predictions != original_neural_network_scores)
# Get the events on which the two techniques disagree (indices are wrapped in single-element arrays)
disagreement_events = [events[index[0]] for index in disagreement_indices]
# Iterate over the events with corresponding network predictions and ground truths, printing information
for event, network_prediction, ground_truth in zip(disagreement_events, network_predictions, ground_truths):
    # Print several attributes of the event that make it identifiable
    print('Unique index:', event.unique_bubble_index)
    print('Date:', event.date)
    print('Run number:', event.run_number)
    print('Event number:', event.event_number)
    print('Run type:', event.run_type)
    # Print out the raw banded Fourier transform data originating in the ROOT file, for the time bin and piezos used for training
    print('Banded frequency domain (raw):')
    print(event.banded_frequency_domain_raw[1:, :, 2])
    # Print the position of the bubble (calculated using the camera)
    print(f'Position: ({event.x_position}, {event.y_position}, {event.z_position})')
    # Print whether the network is correct or the acoustic parameter is correct
    correct = 'Neural network' if network_prediction == ground_truth else 'Acoustic parameter'
    print(correct, 'is correct')
    # Print a blank line for separation
    print()
# Print the total number of events on which the techniques disagree
print(f'{len(disagreement_events)} disagreements in total')
