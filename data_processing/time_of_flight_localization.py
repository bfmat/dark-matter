"""Code for localizing the position of bubbles based on the times of arrival of the signal at multiple piezos"""
# Created by Brendon Matusch, July 2018

import autograd.numpy as np

# A list of the positions of the piezos in 3D space, using the same units as the bubble positions in the ROOT file
PIEZO_POSITIONS = [
    np.array([5, 5, 0]),
    np.array([-5, -5, 0]),
    np.array([3, -3, 0]),
    np.array([-3, 3, 0])
]

# The speed of sound, in distance units per second, in the medium present in the vessel
SPEED_OF_SOUND = 1


def expected_times_of_flight(bubble_position):
    """Calculate the expected times of flight, from a bubble at a certain point to each of the piezos"""
    # Create an array to hold the expected times of flight
    times_of_flight = np.zeros(len(PIEZO_POSITIONS))
    # Iterate over each of the piezo positions, with a corresponding index
    for index, piezo_position in enumerate(PIEZO_POSITIONS):
        # Calculate the distance between this piezo and the bubble
        distance = np.linalg.norm(bubble_position - piezo_position)
        # Divide the distance by the speed of sound and add it to the array
        times_of_flight[index] = distance / SPEED_OF_SOUND
    # Return the array containing the times of flight for every piezo
    return times_of_flight


def timing_error(bubble_position, piezo_timings):
    """Calculate the mean squared error of the expected timings for a certain bubble positions versus those actually observed"""
    # Get the expected overall times of flight to the piezos for this bubble
    times_of_flight = expected_times_of_flight(bubble_position)
    # Subtract the minimum time of flight from all of them, so that the first signal is at time 0
    expected_timings = times_of_flight - np.min(times_of_flight)
    # Return the mean squared error between the actually observed piezo timings and the timings that would be expected for this bubble
    return np.linalg.norm(expected_timings - piezo_timings)
