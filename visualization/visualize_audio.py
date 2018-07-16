#!/usr/bin/env python3
"""A tool for saving the audio from a bubble event as a graph in the time and frequency domains, as well as an audio clip that can be listened to"""
# Created by Brendon Matusch, June 2018

import itertools
import os
import sys
import time

import matplotlib.pyplot as plt
from scipy.io import wavfile

from data_processing.audio_domain_processing import time_to_frequency_domain
from data_processing.bubble_data_point import load_bubble_audio
from data_processing.event_data_set import EventDataSet
from utilities.verify_arguments import verify_arguments

# The number of samples per second to use when saving the audio as WAV files
SAMPLE_RATE = 25_000

# A path to a binary audio file is required
verify_arguments('binary file path')

# Load the audio file located at the path provided
file_path = os.path.expanduser(sys.argv[1])
bubble_audio_list = load_bubble_audio(None, file_path)
# Otherwise, get the first and only element of the list: a NumPy array containing the audio in time domain
time_domain = bubble_audio_list[0]
# Run a Fourier transform on the audio to get it in the frequency domain
frequency_domain = time_to_frequency_domain(time_domain)

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
