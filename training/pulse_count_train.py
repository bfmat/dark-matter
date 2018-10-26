#!/usr/bin/env python3
"""Train a fully connected neural network on the numbers of pulses for each PMT in the DEAP data"""
# Created by Brendon Matusch, August 2018

import keras.backend as K
import numpy as np

from data_processing.deap_serialization import save_test
from data_processing.load_deap_data import load_real_world_deap_data, load_simulated_deap_data
from models.pulse_count_network import create_model

# The number of events to set aside for validation
VALIDATION_SIZE = 2000


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
    # Calculate and print the numbers of (false and true) (positives and negatives) individually
    print(f'Number of true positives for {set_name} data:', np.sum(np.logical_and(predictions_integer == 1, ground_truths == 1)))
    print(f'Number of true negatives for {set_name} data:', np.sum(np.logical_and(predictions_integer == 0, ground_truths == 0)))
    print(f'Number of false positives for {set_name} data:', np.sum(np.logical_and(predictions_integer == 1, ground_truths == 0)))
    print(f'Number of false negatives for {set_name} data:', np.sum(np.logical_and(predictions_integer == 0, ground_truths == 1)))
    # Save the validation ground truths and floating-point predictions in a JSON file named with the set name
    save_test(ground_truths, predictions, events, epoch, f'pulse_count_{set_name}')


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

    # Create an instance of the neural network model
    model = create_model()
    # Iterate for a certain number of epochs
    for epoch in range(100):
        # Train the model for a single epoch
        model.fit(inputs, ground_truths, validation_data=(validation_inputs, validation_ground_truths))
        # Run predictions on the validation set with the trained model, removing the single-element second axis
        validation_predictions = model.predict(validation_inputs)[:, 0]
        # Evaluate the network's predictions, printing statistics and saving a JSON file
        evaluate_predictions(validation_ground_truths, validation_predictions, validation_events, epoch, set_name='validation')
        # Repeat this process for the dedicated test set
        test_predictions = model.predict(test_inputs)[:, 0]
        evaluate_predictions(test_ground_truths, test_predictions, test_events, epoch, set_name='real_world_test')
    # Once training is done, calculate the derivative of the model's input with respect to the output, and take the absolute value to get the effect (positive or negative)
    derivative = K.function([model.input], [K.abs(K.gradients(model.output, model.input)[0])])
    # Run it on the validation and test data, to calculate how each input affects the output
    # Calculate the mean over the example axis, to get a general idea of which inputs have the most effect
    validation_gradients = np.mean(derivative([validation_inputs])[0], axis=0)
    test_gradients = np.mean(derivative([test_inputs[np.nonzero((test_ground_truths))]])[0], axis=0)
    # Calculate the logarithms so it is more visually obvious which are the most significant
    print('Validation gradients:')
    print(list(np.log10(validation_gradients)))
    print('Test gradients:')
    print(list(np.log10(test_gradients)))
