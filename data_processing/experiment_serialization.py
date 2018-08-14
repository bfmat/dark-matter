"""Functions for serializing and loading information about experimental tests which can be used to create graphs"""
# Created by Brendon Matusch, June 2018

import json
import os
import time
from typing import Tuple, Optional, List

import numpy as np

from data_processing.bubble_data_point import BubbleDataPoint
from data_processing.event_data_set import EventDataSet


def save_test(event_data_set: EventDataSet, validation_ground_truths: np.ndarray, validation_network_outputs: np.ndarray, epoch: Optional[int] = None, prefix: str = '') -> None:
    """Save a validation data set, with corresponding experimental network outputs, in a file"""
    # If there are initial indices in the data set, each event corresponds to multiple inputs and ground truths
    # If this is the case, iterate over the indices of this list, adding copies of validation events to a list
    if event_data_set.validation_initial_input_indices is not None:
        validation_events = []
        # Add the number of ground truths as the last index, so the number of inputs for the last event can be calculated
        indices = event_data_set.validation_initial_input_indices.copy()
        indices.append(len(validation_ground_truths))
        # Do not include the last index when iterating; it will be used to calculate the number of inputs for the last event
        for event_index in range(len(indices) - 1):
            # The number of inputs corresponding to the current event is the difference between the current index and the next index
            num_inputs = indices[event_index + 1] - indices[event_index]
            # Add that number of copies of the validation event to the list, so they will correspond correctly with the inputs and ground truths
            validation_events += [event_data_set.validation_events[event_index]] * num_inputs
    # If there are no indices, there is one input per event, so the validation events can be used directly
    else:
        validation_events = event_data_set.validation_events
    # Iterate over tuples of validation events alongside ground truths and network outputs, processing them and adding them to a list
    output_list = []
    for event, ground_truth, network_output in zip(validation_events, validation_ground_truths.tolist(), validation_network_outputs.tolist()):
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
    # Create the folders referenced in the prefix if they are not already present
    os.makedirs(os.path.expanduser(f'~/{prefix}'), exist_ok=True)
    # Create a JSON file in the temporary folder, named with the prefix, current Unix time, and epoch number, save the data in it, and notify the user
    json_file_path = os.path.expanduser(f'~/{prefix}/time{int(time.time())}_epoch{epoch}.json')
    with open(json_file_path, 'w') as output_file:
        json.dump(output_list, output_file)
    print('Data saved at', json_file_path)


def load_test(json_file_path: str) -> Tuple[List[BubbleDataPoint], np.ndarray, np.ndarray]:
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
    events = EventDataSet.load_specific_indices(unique_bubble_indices)
    # Return the list of events, and the ground truths and network outputs as NumPy arrays
    return events, np.array(ground_truths), np.array(network_outputs)
