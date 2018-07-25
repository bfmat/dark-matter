#!/usr/bin/env python3
"""Training script for a neural network that classifies images of bubbles into alpha particles and neutrons"""
# Created by Brendon Matusch, June 2018

import os

from data_processing.event_data_set import EventDataSet, RunType
from data_processing.bubble_data_point import load_bubble_images
from data_processing.experiment_serialization import save_test
from models.image_classification_network import create_model

# Load the event data set from the file, removing multiple-bubble events, disabling acoustic parameter cuts, and keeping background radiation and calibration runs
event_data_set = EventDataSet(
    keep_run_types={
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
    },
    use_wall_cuts=True
)
# Load the image data as NumPy arrays
training_inputs, training_ground_truths, validation_inputs, validation_ground_truths = \
    event_data_set.audio_alpha_classification(
        loading_function=load_bubble_images,
        include_positions=False
    )

# Create an instance of the image processing convolutional neural network
model = create_model()

# Iterate over training and validation for several epochs
for epoch in range(20):
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
    # # Run predictions on the validation data set, and save the experimental run
    # validation_network_outputs = model.predict(validation_inputs)
    # save_test(
    #     event_data_set,
    #     validation_ground_truths,
    #     validation_network_outputs,
    #     epoch=epoch,
    #     prefix='image_'
    # )
    # # Save the current model, named with the epoch number
    # model_path = os.path.expanduser(f'~/image_epoch{epoch}.h5')
    # model.save(model_path)
