#!/usr/bin/env python3
"""Train a fully connected neural network on the numbers of pulses for each PMT in the DEAP data"""
# Created by Brendon Matusch, August 2018

import keras.backend as K
import numpy as np

from data_processing.deap_serialization import save_test
from data_processing.load_deap_data import load_real_world_deap_data, load_simulated_deap_data
from models.pulse_count_network import create_model

# The number of events to set aside for validation
VALIDATION_SIZE = 500

# The number of epochs to train for
EPOCHS = 100


def prepare_events(true_events, false_events):
    """Given 2 lists of events, 1 for each possible ground truth value, produce arrays of inputs and ground truths for training, validation, or testing"""
    # Combine the lists of events into one before converting the data
    events = true_events + false_events
    # Convert the pulse counts from both event lists into a NumPy array of training inputs
    inputs = np.stack([event[0] for event in events])
    # Create a corresponding list of ground truths
    ground_truths = np.array([True] * len(true_events) + [False] * len(false_events))
    # Create a random permutation with the number of inputs and ground truths
    permutation = np.random.permutation(inputs.shape[0])
    # Randomize the inputs and ground truths with the same permutation (otherwise, the validation split would take the end of the arrays)
    inputs = inputs[permutation]
    ground_truths = ground_truths[permutation]
    # Reorder the events accordingly so they can be returned alongside the inputs and ground truths
    events = [events[event_index] for event_index in permutation]
    # Return the arrays of inputs and ground truths, and the list of events
    return inputs, ground_truths, events


def evaluate_predictions(ground_truths: np.ndarray, predictions: np.ndarray, events, epoch: int, set_name: str) -> None:
    """Given arrays of ground truths and corresponding predictions, and a list of corresponding events, print statistics about true and false positives and negatives and save a corresponding JSON file"""
    # Round the predictions to binary values
    predictions_integer = predictions >= 0.5
    # Calculate and return the numbers of (false and true) (positives and negatives) individually
    return np.array([
        np.sum(np.logical_and(predictions_integer == 1, ground_truths == 1)),
        np.sum(np.logical_and(predictions_integer == 0, ground_truths == 0)),
        np.sum(np.logical_and(predictions_integer == 1, ground_truths == 0)),
        np.sum(np.logical_and(predictions_integer == 0, ground_truths == 1))
    ])


# Execute the script only if this is run, not imported
if __name__ == '__main__':
    # Load all simulated events from the file
    neck_events, non_neck_events = load_simulated_deap_data()
    # Convert them to NumPy arrays for training (also getting the reordered list of events)
    inputs, ground_truths, events = prepare_events(neck_events, non_neck_events)
    # Split the inputs and ground truths into training and validation sets
    validation_inputs, training_inputs = np.split(inputs, [VALIDATION_SIZE])
    validation_ground_truths, training_ground_truths = np.split(ground_truths, [VALIDATION_SIZE])
    # Split the events correspondingly (NumPy cannot be used on a list)
    # Take only the validation events, which are located at the beginning of the list
    validation_events = events[:VALIDATION_SIZE]

    # Load all real-world test events from the file
    real_world_neck_events, real_world_neutron_events = load_real_world_deap_data()
    # Prepare the input and ground truth data for testing
    test_inputs, test_ground_truths, test_events = prepare_events(real_world_neck_events, real_world_neutron_events)

    # Create a list to hold the numbers of (false and true) (positives and negatives) for each training run
    performance_statistics = []
    # Train the network multiple times to get an idea of the general accuracy
    for _ in range(3):
        # Create an instance of the neural network model
        model = create_model()
        # Iterate for a certain number of epochs
        for epoch in range(EPOCHS):
            # Print out the epoch number (the fit function does not)
            print('Epoch', epoch)
            # Train the model for a single epoch
            model.fit(training_inputs, training_ground_truths, validation_data=(validation_inputs, validation_ground_truths), class_weight={0: 0.015, 1: 1.0})
            # Run predictions on the validation set with the trained model, removing the single-element second axis
            validation_predictions = model.predict(validation_inputs)[:, 0]
            # Evaluate the network's predictions and add the statistics to the list, only if we are in the last few epochs (we don't care about the other ones, it is still learning then)
            if epoch >= EPOCHS - 10:
                performance_statistics.append(evaluate_predictions(validation_ground_truths, validation_predictions, validation_events, epoch, set_name='validation'))
            # Repeat this process for the dedicated test set
            test_predictions = model.predict(test_inputs)[:, 0]
            evaluate_predictions(test_ground_truths, test_predictions, test_events, epoch, set_name='real_world_test')
    # Add up each of the statistics for the last few epochs and calculate the mean
    statistics_mean = np.mean(np.stack(performance_statistics, axis=0), axis=0)
    # Using these values, calculate and print the percentage of neck alphas removed, and the percentage of nuclear recoils incorrectly removed alongside them
    true_positives, true_negatives, false_positives, false_negatives = statistics_mean
    neck_alphas_removed = true_positives / (true_positives + false_negatives)
    nuclear_recoils_removed = false_positives / (false_positives + true_negatives)
    print('Neck alphas removed:', neck_alphas_removed)
    print('Nuclear recoils removed:', nuclear_recoils_removed)
