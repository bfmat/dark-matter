#!/usr/bin/env python3
"""A tool for graphing variables used to discriminate between wall events and bulk events, and to compare them to the fiducial cuts"""
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
# Get the AP12 and pressure transducer values without correction, for plotting
ap12_values = [event.acoustic_parameter_12 for event in events]
pressure_values = [event.pressure_not_position_corrected for event in events]
# Color the points based on whether or not they pass the fiducial cuts
fiducial_cut_colors = [
    'b' if EventDataSet.passes_fiducial_cuts(event) else 'r'
    for event in events
]

# Scatter plot the AP12 against the pressure transducer data
ax = plt.gca()
ax.scatter(
    x=ap12_values,
    y=pressure_values,
    c=fiducial_cut_colors
)
# Use logarithmic scales for both axes
ax.set_xscale('log')
ax.set_yscale('log')
# Display the graph on screen
plt.show()
