#!/usr/bin/env python3
"""Train a fully connected neural network on the numbers of pulses for each PMT in the DEAP data"""
# Created by Brendon Matusch, August 2018

import numpy as np

from data_processing.load_deap_data import load_deap_data
from models.pulse_count_network import create_model

# The number of events to set aside for validation
VALIDATION_SIZE = 2000

# Load all events from the file
neck_events, non_neck_events = load_deap_data()
# Combine the pulse counts from both event lists into one NumPy array of training inputs
inputs = np.stack([event[0] for event in neck_events + non_neck_events])
# Create a corresponding list of ground truths, using True for neck events and False for non-neck events
ground_truths = np.array([True] * len(neck_events) + [False] * len(non_neck_events))
# Create a random permutation with the number of inputs and ground truths
permutation = np.random.permutation(inputs.shape[0])
# Randomize the inputs and ground truths with the same permutation (otherwise, the validation split would take the end of the arrays)
inputs = inputs[permutation]
ground_truths = ground_truths[permutation]
# Split the inputs and ground truths into training and validation sets
validation_inputs, training_inputs = np.split(inputs, [VALIDATION_SIZE])
validation_ground_truths, training_ground_truths = np.split(ground_truths, [VALIDATION_SIZE])
# Create an instance of the neural network model
model = create_model()
# Iterate for a certain number of epochs
for epoch in range(100):
    # Train the model for a single epoch
    model.fit(inputs, ground_truths, validation_data=(validation_inputs, validation_ground_truths))
    # Run predictions on the validation set with the trained model, removing the single-element second axis and rounding to integers
    validation_predictions = np.rint(model.predict(validation_inputs)[:, 0])
    # Calculate and print the numbers of (false and true) (positives and negatives) individually
    print('Number of true positives:', np.sum(np.logical_and(validation_predictions == 1, validation_ground_truths == 1)))
    print('Number of true negatives:', np.sum(np.logical_and(validation_predictions == 0, validation_ground_truths == 0)))
    print('Number of false positives:', np.sum(np.logical_and(validation_predictions == 1, validation_ground_truths == 0)))
    print('Number of false negatives:', np.sum(np.logical_and(validation_predictions == 0, validation_ground_truths == 1)))
