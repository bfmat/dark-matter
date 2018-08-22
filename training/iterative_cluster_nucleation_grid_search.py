#!/usr/bin/env python3
"""A script for training iterative cluster nucleation with many different combinations of hyperparameters"""
# Created by Brendon Matusch, July 2018

import copy
import os

import numpy as np

# Use only the CPU; it is faster
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras.backend as K
from keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Sequential
from keras.regularizers import l2

from data_processing.bubble_data_point import RunType
from data_processing.event_data_set import EventDataSet
from data_processing.experiment_serialization import save_test

# Iterate over possible configurations for the number of initial training examples, the initial threshold, the threshold multiplier, the L2 lambda, and dropout regularization
for initial_training_examples in [256]:
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
                        print('Initial training examples:', initial_training_examples)
                        print('Initial threshold', initial_threshold)
                        print('Threshold multiplier', threshold_multiplier)
                        print('L2 lambda:', l2_lambda)
                        print('Dropout:', dropout)
                        print('Configuration test index:', configuration_test_index)
                        # Create a description folder path that is used when saving validation data
                        description = f'icn_grid_search/dropout{dropout}/l2_lambda{l2_lambda}/initial_examples{initial_training_examples}/initial_threshold{initial_threshold}/threshold_multiplier{threshold_multiplier}/configuration_test{configuration_test_index}'

                        # Make a mutable copy of the training threshold
                        training_threshold = initial_threshold
                        # Create a data set, running fiducial cuts for the most reasonable data
                        event_data_set = EventDataSet(
                            keep_run_types={
                                RunType.LOW_BACKGROUND,
                                RunType.AMERICIUM_BERYLLIUM,
                                RunType.CALIFORNIUM
                            },
                            use_wall_cuts=True
                        )
                        # Iterate over every training and validation event and remove the images, waveforms, and full resolution Fourier transforms
                        for bubble in event_data_set.training_events + event_data_set.validation_events:
                            bubble.waveform = None
                            bubble.full_resolution_frequency_domain = None
                            bubble.images = None
                        # Make a copy of the full training set to get examples from later
                        original_training_events = event_data_set.training_events.copy()
                        # Truncate the list to only a certain number of initial training examples
                        event_data_set.training_events = event_data_set.training_events[:initial_training_examples]
                        # Remove the actual training events from the list of original training events (that list will be picked from for new examples)
                        original_training_events = [
                            event for event in original_training_events
                            if event not in event_data_set.training_events
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
                        for iteration in range(600):
                            # Output the number of examples there are in the training set for this epoch
                            print(len(event_data_set.training_events),
                                  'training examples for iteration', iteration)
                            # Get the banded frequency domain data and corresponding binary ground truths
                            training_input, training_ground_truths, validation_input, validation_ground_truths = event_data_set.banded_frequency_alpha_classification()

                            # Train the model for a certain number of epochs
                            model.fit(
                                x=training_input,
                                y=training_ground_truths,
                                validation_data=(validation_input, validation_ground_truths),
                                epochs=30
                            )
                            # Run predictions on the validation data set, and save the experimental run
                            validation_network_outputs = model.predict(validation_input)
                            save_test(
                                event_data_set,
                                validation_ground_truths,
                                validation_network_outputs,
                                epoch=iteration,
                                prefix=description
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
                            # Create accumulators to record how many examples were added, and how many are correct
                            examples_added = 0
                            examples_correct = 0
                            # Iterate over the original training events with corresponding predictions
                            for event, prediction in zip(original_training_events, original_training_data_predictions):
                                # If the prediction is within a certain threshold distance of either 0 or 1
                                if min([prediction, 1 - prediction]) < training_threshold:
                                    # Mark the event for removal from the original list
                                    remove_from_original.append(event)
                                    # Copy the event and set its run type so that it is in the corresponding ground truth
                                    bubble_copy = copy.deepcopy(event)
                                    bubble_copy.run_type = RunType.LOW_BACKGROUND if bool(round(prediction)) \
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
                                training_threshold *= threshold_multiplier
                                print(f'Training threshold increased to {training_threshold}')
                            # Otherwise, notify the user what it currently is at
                            else:
                                print(f'Training threshold remains at {training_threshold}')
