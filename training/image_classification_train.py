#!/usr/bin/env python3
"""Training script for a neural network that classifies images of bubbles into alpha particles and neutrons"""
# Created by Brendon Matusch, June 2018

from data_processing.event_data_set import EventDataSet, RunType
from data_processing.bubble_data_point import load_bubble_images
from models.image_classification_network import create_model

# Load the event data set from the file, removing multiple-bubble events, disabling acoustic parameter cuts, and keeping background radiation and calibration runs
event_data_set = EventDataSet(
    filter_multiple_bubbles=True,
    filter_acoustic_parameter=False,
    keep_run_types=set([
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
    ]),
    filter_proportion_randomly=0,
    use_fiducial_cuts=True
)
# Create a training data generator and get validation data array with the image loading function
training_generator_callable, validation_inputs, validation_ground_truths = event_data_set.arbitrary_alpha_classification_generator(
    data_converter=load_bubble_images,
    storage_size=512,
    batch_size=32,
    examples_replaced_per_batch=16
)
training_generator = training_generator_callable()

# Create an instance of the image processing convolutional neural network
model = create_model()

# Iterate over training and validation for 20 epochs
for _ in range(20):
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
    print('Validation loss:', loss)
    print('Validation accuracy:', accuracy)
