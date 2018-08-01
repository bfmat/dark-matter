#!/usr/bin/env python3
"""Training script for a neural network that classifies a higher-resolution frequency domain audio recording into alpha particles and neutrons"""
# Created by Brendon Matusch, July 2018

import os
import sys

from data_processing.event_data_set import EventDataSet
from data_processing.bubble_data_point import RunType, load_bubble_frequency_domain
from data_processing.experiment_serialization import save_test
from models.high_resolution_frequency_network import create_model

# Load the event data set from the file, removing multiple-bubble events, disabling acoustic parameter cuts, and keeping background radiation and calibration runs
event_data_set = EventDataSet(
    keep_run_types={
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
        RunType.CALIFORNIUM
    },
    use_wall_cuts=True
)
# Load training and validation data as NumPy arrays, currying the loading function to disable banding
training_inputs, training_ground_truths, validation_inputs, validation_ground_truths = \
    event_data_set.audio_alpha_classification(
        loading_function=lambda bubble:
        load_bubble_frequency_domain(bubble, banded=False),
        include_positions=False
    )
# Create an instance of the high resolution frequency network
model = create_model()
# Iterate over training and validation for several epochs
for epoch in range(250):
    # Train the model on the input and ground truth arrays
    model.fit(
        x=training_inputs,
        y=training_ground_truths,
        validation_data=(validation_inputs, validation_ground_truths),
        epochs=1
    )
    # Run predictions on the validation data set, and save the experimental run
    validation_network_outputs = model.predict(validation_inputs)
    save_test(
        event_data_set,
        validation_ground_truths,
        validation_network_outputs,
        epoch,
        prefix='high_resolution_ap_simulation_'
    )
