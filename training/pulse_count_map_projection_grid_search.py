#!/usr/bin/env python3
"""Run a grid search for a convolutional neural network trained on the numbers of pulses for each PMT in the DEAP data, projected onto a 2D map"""
# Created by Brendon Matusch, November 2018

import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, InputLayer
from keras.models import Sequential
from keras.regularizers import l2

from data_processing.load_deap_data import load_simulated_deap_data
from data_processing.pmt_map_projection import pmt_map_projection
from training.pulse_count_train import prepare_events, evaluate_predictions

# The number of events to set aside for validation
VALIDATION_SIZE = 500

# The number of epochs to train for
EPOCHS = 100

# Iterate over all hyperparameters that will be searched
for l2_lambda in [0, 0.0003, 0.0006, 0.001, 0.003]:
    for activation in ['tanh']:
        for convolutional_layers in [2, 3]:
            for filters in [8, 16]:
                for dense_layers in [1, 2]:
                    for zero_weight in [0.005, 0.01, 0.015, 0.02]:
                        # Print out the current hyperparameters
                        print('HYPERPARAMETERS')
                        print('L2 Lambda:', l2_lambda)
                        print('Activation:', activation)
                        print('Convolutional Layers:', convolutional_layers)
                        print('Filters:', filters)
                        print('Dense Layers:', dense_layers)
                        print('Zero Weight:', zero_weight)
                        # Train the network multiple times to get an idea of the general accuracy
                        for _ in range(12):
                            # Create a list to hold the numbers of (false and true) (positives and negatives) for this training run
                            performance_statistics = []
                            # Load all simulated events from the file
                            neck_events, non_neck_events = load_simulated_deap_data()
                            # Project the pulse counts onto a map, wrapping in a single-element tuple so the preprocessing function will work
                            neck_events_map, non_neck_events_map = [[(pmt_map_projection(event[0]),) for event in events] for events in [neck_events, non_neck_events]]
                            # Convert them to NumPy arrays for training (also getting the reordered list of events)
                            inputs, ground_truths, events = prepare_events(neck_events_map, non_neck_events_map)
                            # Split the inputs and ground truths into training and validation sets
                            validation_inputs, training_inputs = np.split(inputs, [VALIDATION_SIZE])
                            validation_ground_truths, training_ground_truths = np.split(ground_truths, [VALIDATION_SIZE])
                            # Split the events correspondingly (NumPy cannot be used on a list)
                            # Take only the validation events, which are located at the beginning of the list
                            validation_events = events[:VALIDATION_SIZE]
                            # Create a neural network model that includes several convolutional and dense layers with hyperbolic tangent activations
                            regularizer = l2(l2_lambda)
                            model = Sequential()
                            model.add(InputLayer(input_shape=(10, 35, 1)))
                            model.add(BatchNormalization())
                            # The first convolutional layer is the same no matter what
                            model.add(Conv2D(filters=filters, kernel_size=3, strides=2, activation=activation, kernel_regularizer=regularizer))
                            # The constraints of the input image are tight enough that the individual numbers of layers should probably be handled individually
                            if convolutional_layers == 2:
                                model.add(Conv2D(filters=filters, kernel_size=2, strides=2, activation=activation, kernel_regularizer=regularizer))
                            elif convolutional_layers == 3:
                                model.add(Conv2D(filters=filters, kernel_size=2, strides=1, activation=activation, kernel_regularizer=regularizer))
                                model.add(Conv2D(filters=filters, kernel_size=2, strides=1, activation=activation, kernel_regularizer=regularizer))
                            elif convolutional_layers == 4:
                                model.add(Conv2D(filters=filters, kernel_size=2, strides=1, activation=activation, kernel_regularizer=regularizer))
                                model.add(Conv2D(filters=filters, kernel_size=2, strides=1, activation=activation, kernel_regularizer=regularizer))
                                model.add(Conv2D(filters=filters, kernel_size=2, strides=1, activation=activation, kernel_regularizer=regularizer))
                            model.add(Flatten())
                            if dense_layers == 2:
                                model.add(Dense(64, activation=activation, kernel_regularizer=regularizer))
                            model.add(Dense(16, activation=activation, kernel_regularizer=regularizer))
                            model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizer))
                            # Output a summary of the model's architecture
                            print(model.summary())
                            # Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
                            model.compile(
                                optimizer='adam',
                                loss='mse',
                                metrics=['accuracy']
                            )
                            # Iterate for a certain number of epochs
                            for epoch in range(EPOCHS):
                                # Train the model for a single epoch
                                model.fit(training_inputs, training_ground_truths, validation_data=(validation_inputs, validation_ground_truths), class_weight={0: zero_weight, 1: 1.0})
                                # Run predictions on the validation set with the trained model, removing the single-element second axis
                                validation_predictions = model.predict(validation_inputs)[:, 0]
                                # Evaluate the network's predictions and add the statistics to the list, only if we are in the last few epochs (we don't care about the other ones, it is still learning then)
                                if epoch >= EPOCHS - 10:
                                    performance_statistics.append(evaluate_predictions(validation_ground_truths, validation_predictions, validation_events, epoch, set_name='validation'))
                            # Add up each of the statistics for the last few epochs and calculate the mean
                            statistics_mean = np.mean(np.stack(performance_statistics, axis=0), axis=0)
                            # Using these values, calculate and print the percentage of neck alphas removed, and the percentage of nuclear recoils incorrectly removed alongside them
                            true_positives, true_negatives, false_positives, false_negatives = statistics_mean
                            neck_alphas_removed = true_positives / (true_positives + false_negatives)
                            nuclear_recoils_removed = false_positives / (false_positives + true_negatives)
                            print('Neck alphas removed:', neck_alphas_removed)
                            print('Nuclear recoils removed:', nuclear_recoils_removed)
