#!/usr/bin/env python3
"""Training script for a neural network that classifies audio samples into alpha particles and neutrons"""
# Created by Brendon Matusch, June 2018

from keras.layers import Conv1D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential

from data_processing.event_data_set import EventDataSet
from data_processing.bubble_data_point import RunType, load_bubble_audio

# Load the event data set from the file, removing multiple-bubble events, disabling acoustic parameter cuts, and keeping background radiation and calibration runs
event_data_set = EventDataSet(
    filter_multiple_bubbles=True,
    filter_acoustic_parameter=False,
    keep_run_types=set([
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
    ]),
    filter_proportion_randomly=0.5
)
# Create a training data generator with the audio loading function
training_generator = event_data_set.arbitrary_alpha_classification_generator(
    validation=False,
    data_converter=load_bubble_audio,
    storage_size=512,
    batch_size=32,
    examples_replaced_per_batch=16
)

# Create a one-dimensional convolutional neural network model with hyperbolic tangent activations
# It should take both microphone channels and an entire clip of audio
activation = 'tanh'
model = Sequential([
    InputLayer(input_shape=(250000, 2)),
    Conv1D(filters=16, kernel_size=16, strides=8, activation=activation),
    Dropout(0.25),
    Conv1D(filters=32, kernel_size=12, strides=6, activation=activation),
    Dropout(0.25),
    Conv1D(filters=64, kernel_size=8, strides=4, activation=activation),
    Dropout(0.25),
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
    epochs=20
)
