#!/usr/bin/env python3
"""A tool for graphing the physical positions of events, and comparing them to the fiducial and audio-based wall cuts"""
# Created by Brendon Matusch, July 2018

import matplotlib.pyplot as plt

from data_processing.event_data_set import EventDataSet

# Load all events from the file
events = EventDataSet.load_data_from_file()
# Run all of the standard quality cuts
events = [
    event for event in events
    if EventDataSet.passes_standard_cuts(event)
]
# Get the X and Y positions of the events
x_positions = [event.x_position for event in events]
y_positions = [event.y_position for event in events]
# Color the points based on whether or not they pass the fiducial cuts
fiducial_cut_colors = [
    'b' if EventDataSet.passes_fiducial_cuts(event) else 'r'
    for event in events
]
# Give the points different shapes depending on whether they pass the audio-based wall cuts
audio_cut_shapes = [
    'o' if EventDataSet.passes_audio_wall_cuts(event) else 'x'
    for event in events
]

# Iterate over the attributes together; this must be done because Matplotlib will not accept a list of marker types
for x, y, color, shape in zip(x_positions, y_positions, fiducial_cut_colors, audio_cut_shapes):
    # Scatter plot the position with the corresponding color and shape
    plt.scatter(
        x=x,
        y=y,
        c=color,
        marker=shape
    )
# Label the X and Y axes
plt.xlabel('X Position')
plt.ylabel('Y Position')
# Display the graph on screen
plt.show()
