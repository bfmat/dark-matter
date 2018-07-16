#!/usr/bin/env python3
"""Training script for a convolutional neural network that classifies audio samples into alpha particles and neutrons based on the raw audio waveform"""
# Created by Brendon Matusch, June 2018

import os
import sys

from data_processing.event_data_set import EventDataSet
from data_processing.bubble_data_point import RunType, load_bubble_audio
from data_processing.experiment_serialization import save_test
from models.very_deep_convolutional_network import create_model

# Load a data set from the file, including fiducial cuts
event_data_set = EventDataSet({
    RunType.LOW_BACKGROUND,
    RunType.AMERICIUM_BERYLLIUM,
    RunType.CALIFORNIUM
})
# Load training and validation data as NumPy arrays
training_inputs, training_ground_truths, validation_inputs, validation_ground_truths = \
    event_data_set.audio_alpha_classification(
        loading_function=load_bubble_audio
    )
# Create an instance of the fully convolutional network model
model = create_model()
# Iterate over training and validation for several epochs
for epoch in range(40):
    # Train the model on the input and ground truth arrays
    model.fit(
        x=training_inputs,
        y=training_ground_truths,
        epochs=1,
        batch_size=4
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
        prefix='waveform_'
    )
    # Save the current model, named with the epoch number
    model_path = os.path.expanduser(f'~/waveform_epoch{epoch}.h5')
    model.save(model_path)
