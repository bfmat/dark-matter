"""Functions for serializing and loading information about experimental tests which can be used to create graphs"""
# Created by Brendon Matusch, June 2018

import json
import os
import time
from typing import List, Tuple

import numpy as np

from data_processing.event_data_set import EventDataSet


def save_test(event_data_set: EventDataSet, validation_ground_truths: np.ndarray, validation_network_outputs: np.ndarray) -> None:
    """Save a validation data set, with corresponding experimental network outputs, in a file"""
    # Iterate over tuples of validation event classes alongside ground truths and network outputs, processing them and adding them to a list
    output_list = []
    for event, ground_truth, network_output in zip(event_data_set.validation_events, validation_ground_truths.tolist(), validation_network_outputs.tolist()):
        # Combine the date with the unique index, ground truth value, and network output for this bubble in a dictionary
        bubble_information = {
            # A unique index for this bubble that is constant each time the data set is loaded
            'unique_bubble_index': event.unique_bubble_index,
            # The ground truth for the validation set; may be binary or numeric
            'ground_truth': ground_truth,
            # The network's actual prediction; this is a single-element list
            'network_output': network_output[0]
        }
        # Add the processed dictionary to the list
        output_list.append(bubble_information)
    # Create a JSON file in the temporary folder, named with the current Unix time, save the data in it, and notify the user
    json_file_path = f'/tmp/{int(time.time())}.json'
    with open(json_file_path, 'w') as output_file:
        json.dump(output_list, output_file)
    print('Data saved at', json_file_path)


def load_test(json_file_path: str) -> Tuple[EventDataSet, np.ndarray, np.ndarray]:
    """Load a validation data set from a JSON file encoded by the save function; returns the event data set, the validation ground truths, followed by the validation network outputs"""
    # Load the contents of the JSON file from the provided path
    with open(os.path.expanduser(json_file_path)) as input_file:
        input_list = json.load(input_file)
    # Iterate over the list of dictionaries describing the bubbles, adding information to lists
    unique_bubble_indices = []
    ground_truths = []
    network_outputs = []
    for bubble_information in input_list:
        # Get the bubble index, ground truth, and network output, and add them each to the corresponding list
        unique_bubble_indices.append(bubble_information['unique_bubble_index'])
        ground_truths.append(bubble_information['ground_truth'])
        network_outputs.append(bubble_information['network_output'])
    # Load from disk only the bubbles with the provided set of indices
    event_data_set = EventDataSet.load_specific_indices(unique_bubble_indices)
    # Return the event data set, and the ground truths and network outputs as NumPy arrays
    return event_data_set, np.array(ground_truths), np.array(network_outputs)
