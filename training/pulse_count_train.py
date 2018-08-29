#!/usr/bin/env python3
"""Train a fully connected neural network on the numbers of pulses for each PMT in the DEAP data"""
# Created by Brendon Matusch, August 2018

from typing import Tuple

import numpy as np

from data_processing.deap_serialization import save_test
from data_processing.load_deap_data import load_real_world_deap_data, load_simulated_deap_data
from models.pulse_count_network import create_model

# The number of events to set aside for validation
VALIDATION_SIZE = 2000


def prepare_events(true_events, false_events) -> Tuple[np.ndarray, np.ndarray]:
    """Given 2 lists of events, 1 for each possible ground truth value, produce arrays of inputs and ground truths for training, validation, or testing; also run a cut on the number of pulses"""
    # Take the first photon time for each PMT for each event, substituting 0 if the PMT does not receive any signal
    inputs = np.array([[(timings[0] if timings else 0) for timings in event[1]] for event in true_events + false_events])
    # Create a corresponding list of ground truths
    ground_truths = np.array([True] * len(true_events) + [False] * len(false_events))
    # Calculate the total number of pulses for each event by adding up the pulses for each PMT
    pulse_counts = np.array([sum(event[0]) for event in true_events + false_events])
    # Get the indices of the events in which the number of pulses is within the accepted range (the array comes in a single-element tuple)
    indices_in_pulse_range = np.where(np.logical_and((pulse_counts >= 80), (pulse_counts <= 240)))[0]
    # Take only the inputs and ground truths corresponding to these valid indices
    inputs = inputs[indices_in_pulse_range]
    ground_truths = ground_truths[indices_in_pulse_range]
    # Create a random permutation with the number of inputs and ground truths
    permutation = np.random.permutation(inputs.shape[0])
    # Randomize the inputs and ground truths with the same permutation (otherwise, the validation split would take the end of the arrays)
    inputs = inputs[permutation]
    ground_truths = ground_truths[permutation]
    # Return the arrays of inputs and ground truths
    return inputs, ground_truths


def evaluate_predictions(ground_truths: np.ndarray, predictions: np.ndarray, epoch: int, set_name: str) -> None:
    """Given arrays of ground truths and corresponding predictions, print statistics about true and false positives and negatives and save a corresponding JSON file"""
    # Round the predictions to integer (binary) values
    predictions_integer = np.rint(predictions)
    # Calculate and print the numbers of (false and true) (positives and negatives) individually
    print(f'Number of true positives for {set_name} data:', np.sum(np.logical_and(predictions_integer == 1, ground_truths == 1)))
    print(f'Number of true negatives for {set_name} data:', np.sum(np.logical_and(predictions_integer == 0, ground_truths == 0)))
    print(f'Number of false positives for {set_name} data:', np.sum(np.logical_and(predictions_integer == 1, ground_truths == 0)))
    print(f'Number of false negatives for {set_name} data:', np.sum(np.logical_and(predictions_integer == 0, ground_truths == 1)))
    # Save the validation ground truths and floating-point predictions in a JSON file named with the set name
    save_test(ground_truths, predictions, epoch, f'pulse_count_{set_name}')


# Load all simulated events from the file
neck_events, non_neck_events = load_simulated_deap_data()
# Convert them to NumPy arrays for training
inputs, ground_truths = prepare_events(neck_events, non_neck_events)
# Split the inputs and ground truths into training and validation sets
validation_inputs, training_inputs = np.split(inputs, [VALIDATION_SIZE])
validation_ground_truths, training_ground_truths = np.split(ground_truths, [VALIDATION_SIZE])

# Load all real-world test events from the file
real_world_neck_events, real_world_neutron_events = load_real_world_deap_data()
# Convert them to NumPy arrays for testing
test_inputs, test_ground_truths = prepare_events(real_world_neck_events, real_world_neutron_events)

# Create an instance of the neural network model
model = create_model()
# Iterate for a certain number of epochs
for epoch in range(100):
    # Train the model for a single epoch
    model.fit(inputs, ground_truths, validation_data=(validation_inputs, validation_ground_truths))
    # Run predictions on the validation set with the trained model, removing the single-element second axis
    validation_predictions = model.predict(validation_inputs)[:, 0]
    # Evaluate the network's predictions, printing statistics and saving a JSON file
    evaluate_predictions(validation_ground_truths, validation_predictions, epoch, set_name='validation')
    # Repeat this process for the dedicated test set
    test_predictions = model.predict(test_inputs)[:, 0]
    evaluate_predictions(test_ground_truths, test_predictions, epoch, set_name='real_world_test')
