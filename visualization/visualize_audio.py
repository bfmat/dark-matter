#!/usr/bin/env python3
"""A tool for saving the audio from a bubble event as a graph in the time and frequency domains, as well as an audio clip that can be listened to"""
# Created by Brendon Matusch, June 2018

import sys
import itertools

import matplotlib.pyplot as plt
from numpy.fft import fft

from data_processing.bubble_data_point import load_bubble_audio
from data_processing.event_data_set import EventDataSet
from utilities.verify_arguments import verify_arguments

# A unique identifier for a bubble event is required
verify_arguments('unique bubble identifier')

# Load only the bubble event with this specific index
event_data_set = EventDataSet.load_specific_indices([int(sys.argv[1])])
bubble = event_data_set.validation_events[0]
# Try to load the audio corresponding to this bubble; if it is not found, exit with an error message
bubble_audio_list = load_bubble_audio(bubble)
if len(bubble_audio_list) == 0:
    sys.exit('Audio file does not exist, or access is forbidden')
# Otherwise, get the first and only element of the list: a NumPy array containing the audio
bubble_audio_numpy = bubble_audio_list[0]
print(bubble_audio_numpy.shape)

# Iterate over the audio loaded from the provided folder, and corresponding figure indices
# Ignore the sample rate; only take the data array
for (_, audio_recording), figure_number in zip(load_audio(sys.argv[1]), itertools.count()):
    # Create a new window to graph in
    plt.figure(figure_number)
    # Select the first plot in the window with 2 plots and graph the time domain
    plt.subplot(2, 1, 1)
    plt.plot(audio_recording)
    plt.title('Time Domain')
    # Run a Fourier transform on the sample and graph that in the plot below
    audio_recording_frequency = fft(audio_recording)
    plt.subplot(2, 1, 2)
    plt.plot(audio_recording_frequency)
    plt.title('Frequency Domain')

# Display the graph on screen
plt.show()
