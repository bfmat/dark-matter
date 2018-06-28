#!/usr/bin/env python3
"""Training script for a neural network that classifies images of bubbles into alpha particles and neutrons"""
# Created by Brendon Matusch, June 2018

from keras.layers import Conv2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential

from data_processing.event_data_set import EventDataSet, RunType
from data_processing.experiment_serialization import save_test
from data_processing.bubble_data_point import WINDOW_SIDE_LENGTH

# Load the event data set from the file, removing multiple-bubble events, disabling acoustic parameter cuts, and keeping background radiation and calibration runs
event_data_set = EventDataSet(
    filter_multiple_bubbles=True,
    filter_acoustic_parameter=False,
    keep_run_types=set([
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
    ]),
    filter_proportion_randomly=0.8
)
# Get data generators for both training and validation
training_generator = event_data_set.image_alpha_classification_generator(
    validation=False
)
validation_generator = event_data_set.image_alpha_classification_generator(
    validation=True
)

# Create a convolutional neural network model with hyperbolic tangent activations
activation = 'tanh'
model = Sequential([
    InputLayer(input_shape=(WINDOW_SIDE_LENGTH, WINDOW_SIDE_LENGTH, 1)),
    Conv2D(filters=16, kernel_size=4, strides=2, activation=activation),
    Conv2D(filters=32, kernel_size=3, strides=2, activation=activation),
    Conv2D(filters=64, kernel_size=2, strides=1, activation=activation),
    Flatten(),
    Dense(64, activation=activation),
    Dropout(0.5),
    Dense(16, activation=activation),
    Dropout(0.5),
    Dense(1)
])
# Output a summary of the model's architecture
print(model.summary())
# Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

# Train the model on the loaded data set
model.fit_generator(
    training_generator,
    steps_per_epoch=128,
    validation_data=validation_generator,
    validation_steps=8,
    epochs=20
)
