#!/usr/bin/env python3
"""Run a grid search for a fully connected neural network on the numbers of pulses for each PMT in the DEAP data"""
# Created by Brendon Matusch, August 2018

import numpy as np
from keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Sequential
from keras.regularizers import l2

from data_processing.load_deap_data import load_simulated_deap_data
from training.pulse_count_train import evaluate_predictions, prepare_events

# The number of events to set aside for validation
VALIDATION_SIZE = 500

# The number of epochs to train for
EPOCHS = 100

# Iterate over all hyperparameters that will be searched
for l2_lambda in [0, 0.0003, 0.0006, 0.001, 0.003]:
    for dropout in [0, 0.25, 0.5]:
        for hidden_layers in [2, 4, 6]:
            for activation in ['tanh']:
                # Print out the current hyperparameters
                print('HYPERPARAMETERS')
                print('L2 Lambda:', l2_lambda)
                print('Dropout:', dropout)
                print('Hidden Layers:', hidden_layers)
                print('Activation:', activation)
                # Create a list to hold the numbers of (false and true) (positives and negatives) for each training run
                performance_statistics = []
                # Train the network multiple times to get an idea of the general accuracy
                for _ in range(12):
                    # Load all simulated events from the file
                    neck_events, non_neck_events = load_simulated_deap_data()
                    # Convert them to NumPy arrays for training (also getting the reordered list of events)
                    inputs, ground_truths, events = prepare_events(neck_events, non_neck_events)
                    # Split the inputs and ground truths into training and validation sets
                    validation_inputs, training_inputs = np.split(inputs, [VALIDATION_SIZE])
                    validation_ground_truths, training_ground_truths = np.split(ground_truths, [VALIDATION_SIZE])
                    # Split the events correspondingly (NumPy cannot be used on a list)
                    # Take only the validation events, which are located at the beginning of the list
                    validation_events = events[:VALIDATION_SIZE]

                    # Create a neural network model that includes several dense layers with hyperbolic tangent activations
                    regularizer = l2(l2_lambda)
                    model = Sequential()
                    model.add(InputLayer(input_shape=(255,)))
                    model.add(BatchNormalization())
                    # Iterate over the number of hidden layers
                    for layer_index in range(hidden_layers):
                        # Calculate the number of layers to the output layer, so we can take care of certain special cases
                        layers_to_end = hidden_layers - layer_index
                        # Calculate the number of neurons, which should decrease linearly towards the end of the network, and add a dense layer
                        neurons = layers_to_end * 12
                        model.add(Dense(neurons, activation=activation, kernel_regularizer=regularizer))
                        # If we are not at the last dense layer (excepting the output layer, add a dropout layer)
                        if layers_to_end != 1:
                            model.add(Dropout(dropout))
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
                        # Print out the epoch number (the fit function does not)
                        print('Epoch', epoch)
                        # Train the model for a single epoch
                        model.fit(training_inputs, training_ground_truths, validation_data=(validation_inputs, validation_ground_truths), class_weight={0: 0.01, 1: 1.0})
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
