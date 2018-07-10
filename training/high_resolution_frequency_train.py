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
    filter_multiple_bubbles=True,
    keep_run_types=set([
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
        RunType.CALIFORNIUM_40CM,
        RunType.CALIFORNIUM_60CM,
        RunType.BARIUM_100CM,
        RunType.BARIUM_40CM
    ]),
    use_fiducial_cuts=False
)
# If the option "wall" is passed, discriminate between wall and non-wall events; otherwise, use the default
ground_truth = EventDataSet.is_not_wall_event \
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "wall" \
    else None
# Create a training data generator with the frequency domain audio loading function and whatever ground truth function has been chosen
training_generator_callable, validation_inputs, validation_ground_truths = event_data_set.arbitrary_alpha_classification_generator(
    data_converter=load_bubble_frequency_domain,
    storage_size=512,
    batch_size=32,
    examples_replaced_per_batch=16,
    ground_truth=ground_truth
)
training_generator = training_generator_callable()
# Create an instance of the fully convolutional network model
model = create_model()
# Iterate over training and validation for 20 epochs
for epoch in range(20):
    # Train the model on the generator
    model.fit_generator(
        training_generator,
        steps_per_epoch=128,
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
        prefix='high_resolution_frequency_'
    )
    # Save the current model, named with the epoch number
    model_path = os.path.expanduser(f'~/frequency_epoch{epoch}.h5')
    model.save(model_path)
