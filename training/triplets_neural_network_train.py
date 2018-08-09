#!/usr/bin/env python3
"""A script for training a neural network model to separate loud from quiet triplet events"""
# Created by Brendon Matusch, August 2018

import numpy as np
from keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Model, Sequential
from keras.regularizers import l2

from data_processing.load_triplet_classification_data import load_triplet_classification_data

# Load the triplet events
loud_events, quiet_events = load_triplet_classification_data()
# Stack an input array of flattened position-corrected 8-band Fourier transforms for all of the events together
inputs = np.stack([event.banded_frequency_domain[1:, :, 2].flatten() for event in loud_events + quiet_events])
# Make a binary output list, where 1 is loud and 0 is quiet
outputs = np.concatenate([np.ones(len(loud_events)), np.zeros(len(quiet_events))])

# Create a neural network model that includes several dense layers with hyperbolic tangent activations, L2 regularization, and batch normalization
regularizer = l2(0)
dropout = 0
model = Sequential([
    InputLayer(input_shape=(16,)),
    BatchNormalization(),
    Dense(8, activation='tanh', kernel_regularizer=regularizer),
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
# Train the model, setting aside 15% of the data for validation
model.fit(inputs, outputs, epochs=200, validation_split=0.2)
