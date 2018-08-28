"""Functions for serializing and loading information about DEAP experimental tests which can be used to create graphs"""
# Created by Brendon Matusch, August 2018

import json
import os
import time
from typing import Tuple, Optional, List

import numpy as np


def save_test(validation_ground_truths: np.ndarray, validation_network_outputs: np.ndarray, epoch: Optional[int] = None, prefix: str = '') -> None:
    """Save a validation data set, with corresponding experimental network outputs, in a file"""
    # Iterate over tuples of ground truths and network outputs, processing them and adding them to a list
    output_list = []
    for ground_truth, network_output in zip(validation_ground_truths.tolist(), validation_network_outputs.tolist()):
        # Combine the date with the unique index, ground truth value, and network output for this event in a dictionary
        event_information = {
            # The ground truth for the validation set; may be binary or numeric
            'ground_truth': ground_truth,
            # The network's actual prediction; this is a single-element list
            'network_output': network_output[0]
        }
        # Add the processed dictionary to the list
        output_list.append(event_information)
    # Create the folders referenced in the prefix if they are not already present
    os.makedirs(os.path.expanduser(f'~/{prefix}'), exist_ok=True)
    # Create a JSON file in the temporary folder, named with the prefix, current Unix time, and epoch number, save the data in it, and notify the user
    json_file_path = os.path.expanduser(f'~/{prefix}/time{int(time.time())}_epoch{epoch}.json')
    with open(json_file_path, 'w') as output_file:
        json.dump(output_list, output_file)
    print('Data saved at', json_file_path)


def load_test(json_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a validation data set from a JSON file encoded by the save function; returns the event data set, the validation ground truths, followed by the validation network outputs"""
    # Load the contents of the JSON file from the provided path
    with open(os.path.expanduser(json_file_path)) as input_file:
        input_list = json.load(input_file)
    # Iterate over the list of dictionaries describing the events, adding information to lists
    ground_truths = []
    network_outputs = []
    for event_information in input_list:
        # Get the ground truth, and network output, and add them each to the corresponding list
        ground_truths.append(event_information['ground_truth'])
        network_outputs.append(event_information['network_output'])
    # Return the list of events, and the ground truths and network outputs as NumPy arrays
    return np.array(ground_truths), np.array(network_outputs)
