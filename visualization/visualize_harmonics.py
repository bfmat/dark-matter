#!/usr/bin/env python3
"""A script for loading audio from an event, removing lower frequencies so only harmonics are audible, and visualizing and saving it"""
# Created by Brendon Matusch, July 2018

import sys

import numpy as np

from data_processing.audio_domain_processing import time_to_frequency_domain, frequency_to_time_domain
from data_processing.bubble_data_point import load_bubble_audio
from data_processing.event_data_set import EventDataSet
from utilities.verify_arguments import verify_arguments
from visualization.visualize_audio import visualize_and_save_audio

# A unique bubble index is required
verify_arguments('unique bubble index')
# Load that one bubble from the data file
identifier = int(sys.argv[1])
bubble = EventDataSet.load_specific_indices([identifier])[0]
# Get the audio information associated with the bubble (taking the first and only element of the list)
time_domain = load_bubble_audio(bubble)[0]
# Convert it to frequency domain, preserving complex numbers
frequency_domain = time_to_frequency_domain(time_domain)
# Set the beginning of the frequency domain representation to 0, removing the large spike of low frequencies so the higher harmonics are more apparent
frequency_domain[:3000] = 0
# Convert the modified frequency domain back into the time domain
time_domain = frequency_to_time_domain(frequency_domain)
# Visualize and save the time and frequency domains
visualize_and_save_audio(time_domain, frequency_domain)
