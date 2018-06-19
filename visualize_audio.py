#!/usr/bin/env python3
"""A tool for visualizing audio samples in the time and frequency domains"""
# Created by Brendon Matusch, June 2018

import sys
import itertools

import matplotlib.pyplot as plt
from numpy.fft import fft

from load_audio import load_audio
from verify_arguments import verify_arguments

# A folder containing the audio files in WAV format is expected
verify_arguments('WAV audio folder')

# Iterate over the audio loaded from the provided folder, and corresponding figure indices
for audio_recording, figure_number in zip(load_audio(sys.argv[1]), itertools.count()):
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
