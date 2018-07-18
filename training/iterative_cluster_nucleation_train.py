#!/usr/bin/env python3
"""A script for a semi-supervised learning technique in which a network is trained on a small data set and the most confident previously unlabeled examples are added to the training set on each iteration"""
# Created by Brendon Matusch, July 2018

import copy

import numpy as np

from data_processing.bubble_data_point import RunType
from data_processing.event_data_set import EventDataSet
from data_processing.experiment_serialization import save_test
from models.banded_frequency_network import create_model

# The initial number of examples to put in the training set
INITIAL_TRAINING_EXAMPLES = 512

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
# Make a copy of the full training set, and then truncate the actual training set to a predefined length
original_training_events = event_data_set.training_events.copy()
event_data_set.training_events = event_data_set.training_events[:INITIAL_TRAINING_EXAMPLES]
# Remove the actual training events from the list of original training events (that list will be picked from for new examples)
original_training_events = [
    event for event in original_training_events
    if event not in event_data_set.training_events
]
# Create an instance of the fully convolutional network model
model = create_model()
# Run several training iterations, each containing a number of epochs
for iteration in range(100):
    # Output the number of examples there are in the training set for this epoch
    print(len(event_data_set.training_events),
          'training examples for iteration', iteration)
    # Load the training and validation data with the current examples
    training_input, training_ground_truths, validation_input, validation_ground_truths = event_data_set.banded_frequency_alpha_classification()
    # Train the model for a certain number of epochs on the generator
    model.fit(
        x=training_input,
        y=training_ground_truths,
        validation_data=(validation_input, validation_ground_truths),
        epochs=100
    )
    # Run predictions on the validation data set, and save the experimental run
    validation_network_outputs = model.predict(validation_input)
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
        # Get the frequency domain input data from the event, and add a batch axis
        input_data = np.expand_dims(
            np.concatenate([
                event.banded_frequency_domain_raw[1:].flatten(),
                [event.x_position, event.y_position, event.z_position]
            ]),
            axis=0
        )
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
