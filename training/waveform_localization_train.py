#!/usr/bin/env python3
"""A script for training a neural network to predict the position of a bubble based on multiple audio waveforms from different piezos"""
# Created by Brendon Matusch, July 2018

import os
import sys

from data_processing.event_data_set import EventDataSet
from data_processing.bubble_data_point import RunType, load_bubble_audio
from data_processing.experiment_serialization import save_test
from models.waveform_localization_network import create_model

# Load a data set from the file, without fiducial cuts
event_data_set = EventDataSet(
    keep_run_types={
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
        RunType.CALIFORNIUM
    },
    use_wall_cuts=False
)
# Load training and validation data as NumPy arrays
training_inputs, training_ground_truths, validation_inputs, validation_ground_truths = event_data_set.position_from_waveform()

# Create an instance of the waveform localization network
model = create_model()
