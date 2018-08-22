#!/usr/bin/env python3
"""Train a fully connected neural network on the numbers of pulses for each PMT in the DEAP data"""
# Created by Brendon Matusch, August 2018

import numpy as np

from data_processing.load_deap_data import load_deap_data
from models.pulse_count_network import create_model

# Load all events from the file
neck_events, non_neck_events = load_deap_data()
# Combine the pulse counts from both event lists into one NumPy array of training inputs
inputs = np.stack([event[1] for event in neck_events + non_neck_events])
# Create a corresponding list of ground truths, using True for neck events and False for non-neck events
ground_truths = np.array([True] * len(neck_events) + [False] * len(non_neck_events))
# Create a random permutation with the number of inputs and ground truths
permutation = np.random.permutation(inputs.shape[0])
# Randomize the inputs and ground truths with the same permutation (otherwise, the validation split feature in Keras would take the end of the arrays)
inputs = inputs[permutation]
ground_truths = ground_truths[permutation]
# Create an instance of the neural network model
model = create_model()
# Train the model for a certain number of epochs, setting aside a validation set
model.fit(inputs, ground_truths, epochs=100, validation_split=0.2)
