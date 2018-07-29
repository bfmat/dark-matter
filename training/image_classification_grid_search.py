#!/usr/bin/env python3
"""A script to test many different hyperparameter combinations for the convolutional neural network trained on image data"""
# Created by Brendon Matusch, July 2018

import os

# Use only the second GPU for training (GTX 1060)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from keras.layers import Conv2D, Flatten, Dropout, InputLayer, BatchNormalization, Dense
from keras.models import Model, Sequential
from keras.regularizers import l2

from data_processing.bubble_data_point import load_bubble_images, WINDOW_SIDE_LENGTH, START_IMAGE_INDEX, END_IMAGE_INDEX
from data_processing.event_data_set import EventDataSet, RunType
from data_processing.experiment_serialization import save_test
from models.image_classification_network import create_model

# Load the event data set from the file, removing multiple-bubble events, disabling acoustic parameter cuts, and keeping background radiation and calibration runs
event_data_set = EventDataSet(
    keep_run_types={
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
        RunType.CALIFORNIUM
    },
    use_wall_cuts=True
)
# Load the image data as NumPy arrays
training_inputs, training_ground_truths, validation_inputs, validation_ground_truths = \
    event_data_set.audio_alpha_classification(
        loading_function=load_bubble_images,
        include_positions=False
    )

# Try several different values for dropout, L2 lambda, and convolutional layers per group
for dropout in [0, 0.25, 0.5]:
    for l2_lambda in [0.0003, 0.001, 0.003, 0.006, 0.01]:
        for convolutional_layers_per_group in [2, 3, 4]:
            # Print a few blank lines for separation
            for _ in range(3):
                print()
            # Document the current hyperparameter combination
            print('HYPERPARAMETERS')
            print('L2 Lambda:', l2_lambda)
            print('Dense Dropout:', dropout)
            print('Convolutional Layers Per Group:', convolutional_layers_per_group)
            # Create a description string which is used for saving validation sets
            description = f'image_grid_search_l2_lambda{l2_lambda}_dropout{dropout}_convolutional_layers_per_group{convolutional_layers_per_group}'

            # Calculate the number of images there are stacked along the channels axis
            channels = END_IMAGE_INDEX - START_IMAGE_INDEX
            # Create a network with hyperbolic tangent activations, dropout regularization on the fully connected layers, and L2 regularization everywhere
            activation = 'tanh'
            regularizer = l2(l2_lambda)
            model = Sequential()
            model.add(InputLayer(input_shape=(WINDOW_SIDE_LENGTH, WINDOW_SIDE_LENGTH, channels)))
            model.add(BatchNormalization())
            # Add different numbers of convolutional layers depending on the parameter
            # Always include one with a stride of 2
            model.add(Conv2D(filters=32, kernel_size=4, strides=2, activation=activation, kernel_regularizer=regularizer))
            for _ in range(convolutional_layers_per_group - 1):
                model.add(Conv2D(filters=32, kernel_size=4, activation=activation, kernel_regularizer=regularizer))
            for _ in range(convolutional_layers_per_group):
                model.add(Conv2D(filters=64, kernel_size=3, activation=activation, kernel_regularizer=regularizer))
            for _ in range(convolutional_layers_per_group):
                model.add(Conv2D(filters=128, kernel_size=2, activation=activation, kernel_regularizer=regularizer))
            model.add(Flatten())
            model.add(Dense(64, activation=activation, kernel_regularizer=regularizer))
            model.add(Dropout(dropout))
            model.add(Dense(16, activation=activation, kernel_regularizer=regularizer))
            model.add(Dropout(dropout))
            model.add(Dense(1, activation='sigmoid'))
            # Output a summary of the model's architecture
            print(model.summary())
            # Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['accuracy']
            )

            # Iterate over training and validation for several epochs
            for epoch in range(200):
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
                    epoch=epoch,
                    prefix=description
                )
