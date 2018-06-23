#!/usr/bin/env python3
"""Training script for a neural network that is trained on a recording in frequency domain split into discrete bands"""
# Created by brendon-ai, June 2018

from keras.layers import Dense
from keras.models import Sequential

from event_data import EventDataSet

# Load the event data set from the file
event_data_set = EventDataSet()

# Create a neural network model that includes several dense layers with hyperbolic tangent activations
activation = 'tanh'
model = Sequential([
    Dense(10, input_shape=(10,), activation=activation),
    Dense(6, activation=activation),
    Dense(3, activation=activation),
    Dense(1)
])
