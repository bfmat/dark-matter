#!/usr/bin/env python3
"""A script for loading Pickle file containing ROOT data from PICO-60 run 2, loading images and audio, and embedding them in the data so they can be used elsewhere"""
# Created by Brendon Matusch, July 2018

import os
import pickle

from data_processing.bubble_data_point import load_bubble_audio, load_bubble_frequency_domain, load_bubble_images
from data_processing.event_data_set import EventDataSet, RUN_2_PATH

# Path to the output Pickle file containing audio and image data
AUDIO_AND_IMAGE_FILE_PATH = os.path.expanduser('~/run2alldata.pkl')

# Load the original Pickle file for run 2
with open(RUN_2_PATH, 'rb') as pickle_file:
    bubbles = pickle.load(pickle_file)

# Run standard quality cuts on the data for space efficiency
bubbles = [bubble for bubble in bubbles if EventDataSet.passes_standard_cuts(bubble)]

# Iterate over the bubbles, adding data to them in place
for bubble in bubbles:
    # Load the audio waveform, full resolution frequency domain, and images; add them directly as attributes
    bubble.waveform = load_bubble_audio(bubble)
    bubble.full_resolution_frequency_domain = load_bubble_frequency_domain(bubble, banded=False)
    bubble.images = load_bubble_images(bubble)

# Save the list of bubbles, with the extra data included, as a new Pickle file
with open(AUDIO_AND_IMAGE_FILE_PATH, 'wb') as output_file:
    pickle.dump(bubbles, output_file)
