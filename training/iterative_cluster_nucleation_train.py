#!/usr/bin/env python3
"""A script for a semi-supervised learning technique in which a network is trained on a small data set and the most confident previously unlabeled examples are added to the training set on each iteration"""
# Created by Brendon Matusch, July 2018

import copy

import numpy as np

from data_processing.bubble_data_point import RunType, load_bubble_frequency_domain
from data_processing.event_data_set import EventDataSet
from data_processing.experiment_serialization import save_test
from models.high_resolution_frequency_network import create_model

# The initial number of examples to put in the training set
INITIAL_TRAINING_EXAMPLES = 128

# The distance from 0 or 1 an example must be to be added to the training set
training_threshold = 0.02
# The number (slightly greater than 1) that the threshold is multiplied by when no new examples are added
TRAINING_THRESHOLD_MULTIPLIER = 1.025

# Create a data set, running fiducial cuts for the most reasonable data
event_data_set = EventDataSet({
    RunType.LOW_BACKGROUND,
    RunType.AMERICIUM_BERYLLIUM,
    RunType.CALIFORNIUM
})
# Make a copy of the full training set to get examples from later
original_training_events = event_data_set.training_events.copy()
# Truncate the list to only a certain number of initial training examples
event_data_set.training_events = event_data_set.training_events[:INITIAL_TRAINING_EXAMPLES]
# Remove the actual training events from the list of original training events (that list will be picked from for new examples)
original_training_events = [
    event for event in original_training_events
    if event not in event_data_set.training_events
]
# Create an instance of the fully convolutional network model
model = create_model()
# Run several training iterations, each containing a number of epochs
for iteration in range(400):
    # Output the number of examples there are in the training set for this epoch
    print(len(event_data_set.training_events),
          'training examples for iteration', iteration)
    # Load training and validation data as NumPy arrays, currying the loading function to disable banding
    training_inputs, training_ground_truths, validation_inputs, validation_ground_truths = \
        event_data_set.audio_alpha_classification(
            loading_function=lambda bubble:
            load_bubble_frequency_domain(bubble, banded=False),
            include_positions=True
        )
    # Train the model for a certain number of epochs on the generator
    model.fit(
        x=training_inputs,
        y=training_ground_truths,
        validation_data=(validation_inputs, validation_ground_truths),
        epochs=30
    )
    # Run predictions on the validation data set, and save the experimental run
    validation_network_outputs = model.predict(validation_inputs)
    save_test(
        event_data_set,
        validation_ground_truths,
        validation_network_outputs,
        epoch=iteration,
        prefix='iterative_cluster_nucleation_'
    )
    # Create a list to add to of events that have been added to the main training list
    remove_from_original = []
    # Create accumulators to record how many examples were added, and how many are correct
    examples_added = 0
    examples_correct = 0
    # Iterate over the entire list of potential training examples, running predictions
    for event in original_training_events:
        # Try to load the frequency domain audio data for this event
        audio_data = load_bubble_frequency_domain(event, banded=False)
        # If the audio cannot be loaded, skip to the next iterations
        if not audio_data:
            continue
        # Combine it with the position input data from the event, and add a batch axis
        input_data = [
            np.expand_dims(audio_data[0], axis=0),
            np.expand_dims(
                np.array([event.x_position, event.y_position, event.z_position]),
                axis=0
            )
        ]
        # Run a prediction on the audio sample using the existing neural network
        prediction = model.predict(input_data)
        # If the prediction is within a certain threshold distance of either 0 or 1
        if min([prediction, 1 - prediction]) < training_threshold:
            # Mark the event for removal from the original list
            remove_from_original.append(event)
            # Copy the event and set its run type so that it is in the corresponding ground truth
            bubble_copy = copy.deepcopy(event)
            bubble_copy.run_type = RunType.LOW_BACKGROUND if bool(round(prediction[0, 0])) \
                else RunType.AMERICIUM_BERYLLIUM
            # Add the modified bubble to the training list
            event_data_set.training_events.append(bubble_copy)
            # Update the accumulators according to whether or not the ground truth is right
            examples_added += 1
            if (event.run_type == RunType.LOW_BACKGROUND) == (bubble_copy.run_type == RunType.LOW_BACKGROUND):
                examples_correct += 1
    # Remove the events newly added to the training list from the list of original events
    original_training_events = [
        event for event in original_training_events
        if event not in remove_from_original
    ]
    # Notify the user how many were added and how many were correct
    print(f'{examples_added} examples added; {examples_correct} were correct')
    # If no new examples were added, increase the training data threshold and notify the user
    if examples_added == 0:
        training_threshold *= TRAINING_THRESHOLD_MULTIPLIER
        print(f'Training threshold increased to {training_threshold}')
    # Otherwise, notify the user what it currently is at
    else:
        print(f'Training threshold remains at {training_threshold}')
