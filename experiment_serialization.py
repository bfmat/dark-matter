"""Functions for serializing and loading information about experimental tests which can be used to create graphs"""
# Created by Brendon Matusch, June 2018

import datetime
import json
import time

import numpy as np

from event_data import EventDataSet

# A formatting string for putting dates into ISO 8601 format
DATE_FORMAT = '%Y-%m-%d'


def save_test(event_data_set: EventDataSet, validation_ground_truths: np.ndarray, validation_network_outputs: np.ndarray) -> None:
    """Save a validation data set, with corresponding experimental network outputs, in a file"""
    # Iterate over tuples of validation event classes alongside ground truths and network outputs, processing them and adding them to a list
    output_list = []
    for event, ground_truth, network_output in zip(event_data_set.validation_events, validation_ground_truths, validation_network_outputs):
        # Get the dictionary of attributes of the event class
        attributes = event.__dict__
        # Iterate over the attribute names, converting certain non-serializable data types
        for name in attributes:
            # Get the value corresponding to the name; other methods of iterating are unsafe for modification
            value = attributes[name]
            # If the value is a date, format it into an ISO 8601 string
            if isinstance(value, datetime.date):
                attributes[name] = value.strftime(DATE_FORMAT)
            # If it is a NumPy array, convert it to a list
            if isinstance(value, np.ndarray):
                attributes[name] = value.tolist()
        # Add entries to the dictionary for the corresponding ground truth and network output
        # Convert NumPy types (which cannot be serialized) to native Python types
        attributes['ground_truth'] = bool(ground_truth)
        attributes['network_output'] = float(network_output[0])
        # Add the processed dictionary to the list
        output_list.append(attributes)
    # Create a JSON file in the temporary folder, named with the current Unix time, and save the data in it
    with open(f'/tmp/{int(time.time())}.json', 'w') as output_file:
        json.dump(output_list, output_file)
