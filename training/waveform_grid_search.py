#!/usr/bin/env python3
"""A grid search script for the network trained on raw waveform data"""
# Created by Brendon Matusch, July 2018

import os
import sys

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2

from data_processing.event_data_set import EventDataSet
from data_processing.bubble_data_point import RunType, load_bubble_audio
from data_processing.experiment_serialization import save_test

# Load a data set from the file, including fiducial cuts
event_data_set = EventDataSet({
    RunType.LOW_BACKGROUND,
    RunType.AMERICIUM_BERYLLIUM,
    RunType.CALIFORNIUM
})
# Load training and validation data as NumPy arrays
training_inputs, training_ground_truths, validation_inputs, validation_ground_truths = \
    event_data_set.audio_alpha_classification(
        loading_function=load_bubble_audio,
        include_positions=True
    )

# Iterate over possible configurations for L2 and dropout regularization, number of filters for the first layer (which is scaled up for following layers), kernel size, and number of convolutional layers in most groups
for l2_lambda in [0.003, 0.001, 0.0003]:
    for dense_dropout in [0, 0.25, 0.5]:
        for first_layer_filters in [24, 48]:
            for kernel_size in [3, 5]:
                for convolutional_layers_per_group in [3, 6]:
                    # Print a few blank lines for separation
                    for _ in range(3):
                        print()
                    # Document the current hyperparameter combination
                    print('HYPERPARAMETERS')
                    print('L2 Lambda:', l2_lambda)
                    print('Dense Dropout:', dense_dropout)
                    print('Number of Filters on First Layer:', first_layer_filters)
                    print('Kernel Size', kernel_size)
                    print('Convolutional Layers in Most Groups:', convolutional_layers_per_group)
                    # Create a description string which is used for saving validation sets
                    description = f'waveform_grid_search_l2_lambda{l2_lambda}_dense_dropout{dense_dropout}_first_layer_filters{first_layer_filters}_kernel_size{kernel_size}_convolutional_layers_per_group{convolutional_layers_per_group}'

                    # Create a one-dimensional convolutional neural network model with rectified linear activations, using the Keras functional API
                    # It should take both microphone channels and an entire clip of audio, and take the position of the bubble on all 3 axes as a secondary input
                    activation = 'tanh'
                    padding = 'same'
                    regularizer = l2(l2_lambda)
                    audio_inputs = Input((100_000, 2))
                    x = BatchNormalization()(audio_inputs)
                    x = Conv1D(
                        filters=first_layer_filters,
                        kernel_size=80,
                        strides=4,
                        activation=activation,
                        kernel_regularizer=regularizer,
                        padding=padding
                    )(x)
                    x = MaxPooling1D(6)(x)
                    x = BatchNormalization()(x)
                    for _ in range(convolutional_layers_per_group):
                        x = Conv1D(
                            filters=first_layer_filters,
                            kernel_size=kernel_size,
                            activation=activation,
                            kernel_regularizer=regularizer,
                            padding=padding
                        )(x)
                        x = BatchNormalization()(x)
                    x = MaxPooling1D(6)(x)
                    x = BatchNormalization()(x)
                    for _ in range(convolutional_layers_per_group + 1):
                        x = Conv1D(
                            filters=first_layer_filters * 2,
                            kernel_size=kernel_size,
                            activation=activation,
                            kernel_regularizer=regularizer,
                            padding=padding
                        )(x)
                        x = BatchNormalization()(x)
                    x = MaxPooling1D(6)(x)
                    x = BatchNormalization()(x)
                    for _ in range(convolutional_layers_per_group * 2):
                        x = Conv1D(
                            filters=first_layer_filters * 4,
                            kernel_size=kernel_size,
                            activation=activation,
                            kernel_regularizer=regularizer,
                            padding=padding
                        )(x)
                        x = BatchNormalization()(x)
                    x = MaxPooling1D(6)(x)
                    x = BatchNormalization()(x)
                    for _ in range(convolutional_layers_per_group):
                        x = Conv1D(
                            filters=first_layer_filters * 8,
                            kernel_size=kernel_size,
                            activation=activation,
                            kernel_regularizer=regularizer,
                            padding=padding
                        )(x)
                        x = BatchNormalization()(x)
                    x = Flatten()(x)
                    # Create a secondary input for the 3 axes and concatenate it to the outputs of the convolutional layers
                    axes_inputs = Input((3,))
                    x = concatenate([x, axes_inputs])
                    x = BatchNormalization()(x)
                    x = Dense(64, activation=activation, kernel_regularizer=regularizer)(x)
                    x = Dropout(dense_dropout)(x)
                    x = Dense(16, activation=activation, kernel_regularizer=regularizer)(x)
                    x = Dropout(dense_dropout)(x)
                    outputs = Dense(1, activation='sigmoid', kernel_regularizer=regularizer)(x)
                    model = Model(inputs=[audio_inputs, axes_inputs], outputs=outputs)
                    # Output a summary of the model's architecture
                    print(model.summary())
                    # Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
                    model.compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=['accuracy']
                    )

                    # Iterate over training and validation for several epochs
                    for epoch in range(100):
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
                        # Run predictions on the validation data set, and save the experimental run
                        validation_network_outputs = model.predict(validation_inputs)
                        save_test(
                            event_data_set,
                            validation_ground_truths,
                            validation_network_outputs,
                            epoch,
                            prefix=description
                        )
