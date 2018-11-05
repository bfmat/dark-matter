#!/usr/bin/env python3
"""Train a convolutional neural network on the numbers of pulses for each PMT in the DEAP data, projected onto a 2D map"""
# Created by Brendon Matusch, August 2018

import numpy as np
from scipy.misc import imsave

from data_processing.deap_serialization import save_test
from data_processing.load_deap_data import load_real_world_deap_data, load_simulated_deap_data
from data_processing.pmt_map_projection import pmt_map_projection
from models.map_projection_cnn import create_model
from training.pulse_count_train import prepare_events, evaluate_predictions

# The number of events to set aside for validation
VALIDATION_SIZE = 500

# The number of epochs to train for
EPOCHS = 100

# Load all simulated events from the file
neck_events, non_neck_events = load_simulated_deap_data()
# Project the pulse counts onto a map, wrapping in a single-element tuple so the preprocessing function will work
neck_events_map, non_neck_events_map = [[(pmt_map_projection(event[0]),) for event in events] for events in [neck_events, non_neck_events]]
imsave('/home/brendonm/map_projection.png', neck_events_map[0][0][:, :, 0])
# Convert them to NumPy arrays for training (also getting the reordered list of events)
inputs, ground_truths, events = prepare_events(neck_events_map, non_neck_events_map)
# Split the inputs and ground truths into training and validation sets
validation_inputs, training_inputs = np.split(inputs, [VALIDATION_SIZE])
validation_ground_truths, training_ground_truths = np.split(ground_truths, [VALIDATION_SIZE])
# Split the events correspondingly (NumPy cannot be used on a list)
# Take only the validation events, which are located at the beginning of the list
validation_events = events[:VALIDATION_SIZE]

# Load all real-world test events from the file
real_world_neck_events, real_world_neutron_events = load_real_world_deap_data()
# Repeat the map projection with the real-world events
real_neck_events_map, real_neutron_events_map = [[(pmt_map_projection(event[0]),) for event in events] for events in [real_world_neck_events, real_world_neutron_events]]
# Prepare the input and ground truth data for testing
test_inputs, test_ground_truths, test_events = prepare_events(real_neck_events_map, real_neutron_events_map)

# Create a list to hold the numbers of (false and true) (positives and negatives) for each training run
performance_statistics = []
# Train the network multiple times to get an idea of the general accuracy
for _ in range(3):
    # Create an instance of the neural network model
    model = create_model()
    # Iterate for a certain number of epochs
    for epoch in range(EPOCHS):
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
