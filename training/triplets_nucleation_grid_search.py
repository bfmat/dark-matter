#!/usr/bin/env python3
"""A script for training to discriminate between loud and quiet triplet alpha events using iterative cluster nucleation"""
# Created by Brendon Matusch, August 2018

import copy
import os
import random

import numpy as np

# Use only the CPU; it is faster
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Sequential
from keras.regularizers import l2

from data_processing.bubble_data_point import RunType
from data_processing.event_data_set import EventDataSet
from data_processing.experiment_serialization import save_test
from data_processing.load_triplet_classification_data import load_triplet_classification_data

# Number of triplet events to put in the validation set
VALIDATION_SET_SIZE = 8

# Iterate over possible configurations for the number of initial training examples, the initial threshold, the threshold multiplier, the L2 lambda, and dropout regularization
for initial_threshold in [0.01, 0.02]:
    for threshold_multiplier in [1.025, 1.05]:
        for l2_lambda in [0, 0.001, 0.003]:
            for dropout in [0, 0.25, 0.5]:
                # Test each configuration multiple times so the initial set does not cause a bias
                for configuration_test_index in range(3):
                    # Print a few blank lines for separation
                    for _ in range(3):
                        print()
                    # Document the current hyperparameter configuration
                    print('HYPERPARAMETERS')
                    print('Initial threshold', initial_threshold)
                    print('Threshold multiplier', threshold_multiplier)
                    print('L2 lambda:', l2_lambda)
                    print('Dropout:', dropout)
                    print('Configuration test index:', configuration_test_index)
                    # Create a description string that is used when saving validation data
                    description = f'triplets_icn_grid_search_dropout{dropout}_l2_lambda{l2_lambda}_initial_threshold{initial_threshold}_threshold_multiplier{threshold_multiplier}_configuration_test{configuration_test_index}_'

                    # Make a mutable copy of the training threshold
                    training_threshold = initial_threshold

                    # Load all of the events and run standard and wall cuts
                    events = [
                        event for event in EventDataSet.load_data_from_file()
                        if EventDataSet.passes_standard_cuts(event)
                        and EventDataSet.passes_fiducial_cuts(event)
                        and EventDataSet.passes_audio_wall_cuts(event)
                    ]
                    # Iterate over every event and remove the images, waveforms, and full resolution Fourier transforms
                    for bubble in events:
                        bubble.waveform = None
                        bubble.full_resolution_frequency_domain = None
                        bubble.images = None
                    # Randomize the list of events
                    random.shuffle(events)
                    # Load the triplet events
                    loud_events, quiet_events = load_triplet_classification_data()
                    # Combine them into one list
                    triplet_events = loud_events + quiet_events
                    # Initialize a list of training ground truths with the known identities of the triplet events
                    triplet_ground_truths = list(np.concatenate([np.ones(len(loud_events)), np.zeros(len(quiet_events))]))
                    # Generate a random permutation and use it to order the triplet events and ground truths
                    order = np.random.permutation(len(triplet_events))
                    triplet_events = [triplet_events[index] for index in order]
                    triplet_ground_truths = [triplet_ground_truths[index] for index in order]
                    # Split both the events and ground truths into validation and initial training sets
                    validation_events = triplet_events[:VALIDATION_SET_SIZE]
                    training_events = triplet_events[VALIDATION_SET_SIZE:]
                    validation_ground_truths = triplet_ground_truths[:VALIDATION_SET_SIZE]
                    training_ground_truths = triplet_ground_truths[VALIDATION_SET_SIZE:]
                    # Remove the actual training events from the list of original training events (that list will be picked from for new examples)
                    original_training_events = [
                        event for event in events
                        if event not in training_events
                    ]

                    # Create a neural network model that includes several dense layers with hyperbolic tangent activations, L2 regularization, and batch normalization
                    regularizer = l2(l2_lambda)
                    activation = 'tanh'
                    model = Sequential([
                        InputLayer(input_shape=(16,)),
                        BatchNormalization(),
                        Dense(12, activation=activation, kernel_regularizer=regularizer),
                        Dropout(dropout),
                        Dense(8, activation=activation, kernel_regularizer=regularizer),
                        Dropout(dropout),
                        Dense(1, activation='sigmoid', kernel_regularizer=regularizer)
                    ])
                    # Output a summary of the model's architecture
                    print(model.summary())
                    # Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
                    model.compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=['accuracy']
                    )

                    # Run several training iterations, each containing a number of epochs
                    for iteration in range(200):
                        # Output the number of examples there are in the training set for this epoch
                        print(len(training_events), 'training examples for iteration', iteration)
                        # Generate banded Fourier transform inputs from the training and validation events
                        training_inputs, validation_inputs = [
                            np.stack([
                                event.banded_frequency_domain[1:, :, 2].flatten()
                                for event in events_list
                            ])
                            for events_list in [training_events, validation_events]
                        ]

                        # Train the model for a certain number of epochs
                        model.fit(
                            x=training_inputs,
                            y=training_ground_truths,
                            validation_data=(validation_inputs, validation_ground_truths),
                            epochs=30
                        )

                        # If there are no original unlabeled training events left, make the predictions an empty list
                        if not original_training_events:
                            original_training_data_predictions = []
                        # Otherwise, load them and run predictions
                        else:
                            # Get banded frequency data from the original list of potential training events
                            original_training_data = np.stack([
                                event.banded_frequency_domain[1:, :, 2].flatten()
                                for event in original_training_events
                            ])
                            # Run predictions on each of these examples and remove the unnecessary extra axis
                            original_training_data_predictions = model.predict(original_training_data)[:, 0]
                        # Create a list to add to of events that have been added to the main training list
                        remove_from_original = []
                        # Create accumulators to record how many examples were added
                        examples_added = 0
                        # Iterate over the original training events with corresponding predictions
                        for event, prediction in zip(original_training_events, original_training_data_predictions):
                            # If the prediction is within a certain threshold distance of either 0 or 1
                            if min([prediction, 1 - prediction]) < training_threshold:
                                # Mark the event for removal from the original list
                                remove_from_original.append(event)
                                # Add it to the training list
                                training_events.append(event)
                                # Add the classification for this event to the ground truths
                                training_ground_truths.append(bool(round(prediction)))
                        # Remove the events newly added to the training list from the list of original events
                        original_training_events = [
                            event for event in original_training_events
                            if event not in remove_from_original
                        ]
                        # Notify the user how many were added and how many were correct
                        print(f'{examples_added} examples added')
                        # If no new examples were added, increase the training data threshold and notify the user
                        if examples_added == 0:
                            training_threshold *= threshold_multiplier
                            print(f'Training threshold increased to {training_threshold}')
                        # Otherwise, notify the user what it currently is at
                        else:
                            print(f'Training threshold remains at {training_threshold}')
