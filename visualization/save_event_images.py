#!/usr/bin/env python3
"""A script for saving the first list images corresponding to a certain event, tiled together"""
# Created by Brendon Matusch, August 2018

import os
import sys

import numpy as np
from skimage.io import imsave

from data_processing.bubble_data_point import load_bubble_images
from data_processing.event_data_set import EventDataSet
from utilities.verify_arguments import verify_arguments

# A unique bubble index is required
verify_arguments('unique bubble index')
# Load that one bubble from the data file
identifier = int(sys.argv[1])
bubble = EventDataSet.load_specific_indices([identifier])[0]
# Load the images associated with this bubble, taking the first array
images = load_bubble_images(bubble)[0]
# These should be shaped into a 5 by 2 grid of images
# Create such an empty array, using the side length of the square images
window_side_length = images.shape[0]
# Use the same data type as the source images
grid = np.zeros((window_side_length * 5, window_side_length * 2), dtype=images.dtype)
# Iterate over each axis of the grid
for grid_x in range(5):
    for grid_y in range(2):
        # Calculate the corresponding top left corner of the image, using the window side length
        left_x = grid_x * window_side_length
        top_y = grid_y * window_side_length
        # Get the corresponding index in the array of images
        image_index = grid_x + (grid_y * 5)
        # Get the corresponding window and place it in the grid image
        grid[left_x:left_x + window_side_length, top_y:top_y + window_side_length] = images[:, :, image_index]
# Transpose the grid because it is expected in (X, Y) dimension order
grid = np.transpose(grid)
# Save the grid image to disk
imsave(os.path.expanduser('~/image_grid.png'), grid)
