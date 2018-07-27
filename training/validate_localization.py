#!/usr/bin/env python3
"""A script for running time of flight localization on many different examples and comparing it to the positions calculated with the camera"""
# Created by Brendon Matusch, July 2018

import random

import numpy as np
from scipy.optimize import minimize

from data_processing.event_data_set import EventDataSet
from data_processing.time_of_flight_localization import localize_bubble

# Load all events straight from the file, and apply standard quality cuts
events = [
    event for event in EventDataSet.load_data_from_file()
    if EventDataSet.passes_standard_cuts(event)
]
# Randomize the order of the events and take a certain number from the beginning
random.shuffle(events)
events = events[:500]


def mean_squared_positional_error(piezo_positions_combined: np.ndarray) -> float:
    """Run through all events in the data set, approximating their positions using audio and comparing them to those calculated based on the camera, and return the mean squared error over all events"""
    # Reshape the piezo positions into an N x 3 array
    piezo_positions = np.reshape(piezo_positions_combined, (-1, 3))
    # Iterate over all of the events, adding squared errors to a list
    squared_errors = []
    for event in events:
        # Take only the working piezos from the list of zero times
        piezo_time_zero = np.array(event.piezo_time_zero)[[0, 1, 2, 4, 5, 6]]
        # Subtract the first piezo signal time from all of them, so they are either 0 or positive, and at least one is 0
        piezo_timings = piezo_time_zero - np.min(piezo_time_zero)
        # Approximate the position of the bubble based the times audio starts at the different piezos
        audio_position = localize_bubble(piezo_timings, piezo_positions)
        # Subtract it from the position calculated by the camera to get an error vector
        camera_position = np.array([event.x_position, event.y_position, event.z_position])
        error_vector = audio_position - camera_position
        # Square the error values and append them to the list for all events
        squared_error_vector = error_vector ** 2
        squared_errors += squared_error_vector.tolist()
    # Print and return the mean of all of the squared errors
    mean_squared_error = np.mean(squared_errors)
    print(f'Mean squared error over all events is {mean_squared_error}')
    return mean_squared_error


# Train the piezo positions to optimize the mean squared error
minimize(
    fun=mean_squared_positional_error,
    x0=np.random.randn(18),
    method='Nelder-Mead'
).x
