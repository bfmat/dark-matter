#!/usr/bin/env python3
"""Training script for a neural network that is trained on a recording in frequency domain split into discrete bands"""
# Created by brendon-ai, June 2018

from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential

from event_data import EventDataSet

# Load the event data set from the file with the default filters
event_data_set = EventDataSet()
# Get the banded frequency domain data and corresponding binary ground truths
banded_data, alpha_ground_truths = event_data_set.banded_frequency_alpha_classification()

# The input dimension to the network should be the number of values per banded data point
input_dimension = banded_data.shape[1]
# Create a neural network model that includes several dense layers with hyperbolic tangent activations, dropout, and batch normalization
activation = 'tanh'
model = Sequential([
    BatchNormalization(input_shape=(input_dimension,)),
    Dense(input_dimension, activation=activation),
    Dense(72, activation=activation),
    Dropout(0.5),
    Dense(72, activation=activation),
    Dropout(0.5),
    Dense(24, activation=activation),
    Dropout(0.5),
    Dense(6, activation=activation),
    Dropout(0.5),
    Dense(1)
])
# Output a summary of the model's architecture
print(model.summary())
# Use a binary classification loss function and an Adam optimizer, and print the accuracy while training
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model on the loaded data set
model.fit(
    x=banded_data,
    y=alpha_ground_truths,
    epochs=100,
    validation_split=0.1
)
