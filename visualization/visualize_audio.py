#!/usr/bin/env python3
"""A tool for saving the audio from a bubble event as a graph in the time and frequency domains, as well as an audio clip that can be listened to"""
# Created by Brendon Matusch, June 2018

import itertools
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from data_processing.audio_domain_processing import time_to_frequency_domain
from data_processing.bubble_data_point import load_bubble_audio, load_bubble_frequency_domain
from data_processing.event_data_set import EventDataSet
from utilities.verify_arguments import verify_arguments

# The number of samples per second to use when saving the audio as WAV files
SAMPLE_RATE = 25_000


def visualize_and_save_audio(time_domain: np.ndarray, frequency_domain: np.ndarray) -> None:
    """Given time and frequency domain arrays, graph an audio recording and save it in an audible format"""
    # Iterate over the channel indices, which correspond to the row indices in the graph
    channels = time_domain.shape[1]
    for channel_index in range(channels):
        # Select the plot in the first column and whatever row we are on, and graph the time domain
        time_domain_plot_index = (channel_index * 2) + 1
        time_domain_channel = time_domain[:, channel_index]
        plt.subplot(channels, 2, time_domain_plot_index)
        plt.plot(time_domain_channel)
        plt.title(f'Time Domain, Channel {channel_index}')
        # Graph the audio in frequency domain in the plot to the right
        frequency_domain_plot_index = time_domain_plot_index + 1
        frequency_domain_channel = frequency_domain[:, channel_index]
        plt.subplot(channels, 2, frequency_domain_plot_index)
        plt.plot(frequency_domain_channel)
        plt.title(f'Frequency Domain, Channel {channel_index}')

        # Take the time domain for this channel, save it as a WAV file, and notify the user
        file_path = os.path.expanduser(f'~/channel_{channel_index}.wav')
        wavfile.write(file_path, SAMPLE_RATE, time_domain_channel)
        print(f'Saved audio for channel {channel_index} as {file_path}')

    # Show the graph to the user
    plt.show()


# Load and visualize audio only if this script is being executed directly and not imported
if __name__ == '__main__':
    # A unique bubble index is required
    verify_arguments('unique bubble index')
    # Load that one bubble from the data file
    identifier = int(sys.argv[1])
    bubble = EventDataSet.load_specific_indices([identifier])[0]
    # Get the audio information associated with the bubble (taking the first and only element of the list)
    time_domain = load_bubble_audio(bubble)[0]
    # Also, get the flattened full-resolution Fourier transform in the same way
    frequency_domain_flat = load_bubble_frequency_domain(bubble, banded=False)[0]
    # Reshape the frequency domain into an array with 2 channels
    frequency_domain = np.reshape(frequency_domain_flat, (-1, 2))
    # Visualize and save the time and frequency domains
    visualize_and_save_audio(time_domain, frequency_domain)
