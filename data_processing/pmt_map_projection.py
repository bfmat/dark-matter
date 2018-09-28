"""A system for projecting 3D PMT pulse count data onto a 2D map using an approximate Mercator map projection"""
# Created by Brendon Matusch, September 2018

import numpy as np
from scipy.ndimage import zoom

from data_processing.pmt_positions import X_POSITIONS, Y_POSITIONS, Z_POSITIONS


def pmt_map_projection(pulse_counts: np.ndarray) -> np.ndarray:
    """Project a list of pulse counts, at predefined positions along the sphere, onto a 2D map"""
    # Get the unique Z positions (the PMTs are arranged into circular rows)
    row_positions = np.unique(Z_POSITIONS)
    # Create a list of lists which will contain tuples of the form (pulse count, X, Y) for each row individually
    values_by_row = []
    # Add empty lists to it, one for each of the rows
    for _ in range(len(row_positions)):
        values_by_row.append([])
    # Iterate over the pulse counts alongside positions in 3D space
    for pulse_count, x_position, y_position, z_position in zip(pulse_counts, X_POSITIONS, Y_POSITIONS, Z_POSITIONS):
        # Get the index of the row that this position belongs to (which corresponds to the Y position in the resulting map)
        row_index = np.where(row_positions == z_position)[0][0]
        # Add this data point to the list for the corresponding row
        values_by_row[row_index].append((pulse_count, x_position, y_position))
    # Create an empty image where the height is the number of rows and the width is the largest number of PMTs in a row (the rest will be scaled accordingly)
    largest_row_size = max(len(row) for row in values_by_row)
    map_image = np.zeros(shape=(largest_row_size, len(values_by_row)), dtype=int)
    # Iterate over the rows with corresponding indices, adding them to the map projection
    for row_index, row in enumerate(values_by_row):
        # Get the angle around the Z axis (in radians) of each of the data points in this row (by representing the vector as a complex number)
        angles = [np.angle(x_position + (y_position * 1j)) for _, x_position, y_position in row]
        # Get the indices to sort the list of angles (and correspondingly, the data points)
        sorted_indices = np.argsort(angles)
        # Get the pulse counts and sort them according to the angles
        sorted_pulse_counts = np.array([pulse_count for pulse_count, _, _ in row])[sorted_indices]
        # If the number of pulse counts in this row is equal to the maximum, go ahead and set the row in the image
        if len(sorted_pulse_counts) == largest_row_size:
            map_image[:, row_index] = sorted_pulse_counts
        # Otherwise, use an image scaling algorithm to resize it to fit
        else:
            map_image[:, row_index] = zoom(sorted_pulse_counts, largest_row_size / len(sorted_pulse_counts))
    # Add an extra batch dimension with only 1 element, so Keras convolutional layers will accept it
    map_image = np.expand_dims(map_image, axis=-1)
    # Return the map with all of the data projected onto it
    return map_image
