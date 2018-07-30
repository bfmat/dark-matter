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
    use_wall_cuts=False,
    use_run_1=True
)
# Load training and validation data as NumPy arrays
training_inputs, training_ground_truths, validation_inputs, validation_ground_truths = event_data_set.position_from_waveform()

# Create an instance of the waveform localization network
model = create_model()

# Iterate over training and validation for several epochs
for epoch in range(100):
    # Train the model on the input and ground truth arrays
    model.fit(
        x=training_inputs,
        y=training_ground_truths,
        epochs=1
    )
    # Evaluate the model on the validation data set
    loss, accuracy = model.evaluate(
        x=validation_inputs,
        y=validation_ground_truths,
        verbose=0
    )
    # Output the validation loss and accuracy to the user
    print('Validation loss:', loss)
    print('Validation accuracy:', accuracy)
    # Run predictions on the validation data set, and save the experimental run
    validation_network_outputs = model.predict(validation_inputs)
    save_test(
        event_data_set,
        validation_ground_truths,
        validation_network_outputs,
        epoch,
        prefix='waveform_localization_'
    )
    # Save the current model, named with the epoch number
    model_path = os.path.expanduser(f'~/waveform_localization_epoch{epoch}.h5')
    model.save(model_path)
