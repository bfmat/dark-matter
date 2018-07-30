#!/usr/bin/env python3
"""A script for a semi-supervised training technique where certain examples have defined ground truths and others are assigned probabilistic ground truths based on what they are predicted to be by the network"""
# Created by Brendon Matusch, July 2018

import copy

import numpy as np
from keras.optimizers import SGD

from data_processing.bubble_data_point import RunType
from data_processing.event_data_set import EventDataSet
from data_processing.experiment_serialization import save_test
from models.banded_frequency_network import create_model

# The number of training examples for which the ground truth is actually used, and is not dynamically generated
DEFINITIVE_TRAINING_EXAMPLES = 128

# The value (greater than 1) to add to the gravity multiplier every epoch
GRAVITY_MULTIPLIER_INCREMENT = 0.003

# The root to use to flatten out the middle of the gravity function of the prediction spectrum
DISTORTION_ROOT = 9

# Create an instance of the fully connected neural network
model = create_model()
# Recompile the model to use a simple stochastic gradient descent optimizer without any momentum or Nesterov; the ground truth tweaking in this system should not be combined with a more complex optimizer, which will interfere with the desired effects
model.compile(
    optimizer=SGD(),
    loss='mse',
    metrics=['accuracy']
)

# Create a data set, running fiducial cuts for the most reasonable data
event_data_set = EventDataSet(
    keep_run_types={
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
        RunType.CALIFORNIUM
    },
    use_wall_cuts=True
)
# Get the banded frequency domain data for training and validation
training_input, training_ground_truths, validation_input, validation_ground_truths = event_data_set.banded_frequency_alpha_classification()
# For comparison later, store the original training ground truths of the examples for which they will be changed
original_training_ground_truths = training_ground_truths[DEFINITIVE_TRAINING_EXAMPLES:]
# Make a copy of the event data set, replacing the validation set with the corresponding training examples for saving the training data
training_data_set = copy.deepcopy(event_data_set)
training_data_set.validation_events = \
    event_data_set.training_events[DEFINITIVE_TRAINING_EXAMPLES:]
# Initially, set all of the training ground truths (except for the few for which the original ground truth is kept) to 0.5
# This keeps the network from learning anything it shouldn't until at least some training has been done on the definitive data
# It must first be converted to floating-point
training_ground_truths = training_ground_truths.astype(float)
training_ground_truths[DEFINITIVE_TRAINING_EXAMPLES:] = 0.5


def gravitational_ground_truth_offsets(predictions: np.ndarray, distortion_root: float, gravity_multiplier: float) -> np.ndarray:
    """Get an array of ground truth offsets based on a gravitational model where examples classified very close to one edge will be pulled toward that edge, and examples near the middle will make little difference"""
    # The function should pass through (0.5, 0), should change very little near that point, and should rapidly asymptote in the negative or positive directions as the prediction comes close to 0 or 1
    # First, scale the predictions to the range of -1 to 1
    predictions_scaled = (predictions - 0.5) * 2
    # Take the hyperbolic tangent so examples in the middle are affected minimally
    hyperbolic_tangent = np.tanh(predictions_scaled)
    # Take the Nth root (removing the sign and multiplying it back in after so the negative side is the same as the positive side) of the hyperbolic tangent so that the area around 0 is squashed
    root_distorted = np.sign(hyperbolic_tangent) * np.power(np.abs(hyperbolic_tangent), distortion_root)
    # Multiply it by a constant so the gravitational offset does not dominate the training process
    return root_distorted * gravity_multiplier


# The gravity multiplier should start at 0 and is added to every epoch
gravity_multiplier = 0

# Iterate for a certain number of epochs
for epoch in range(1000):
    # Train the model for one epoch
    model.fit(
        x=training_input,
        y=training_ground_truths,
        validation_data=(validation_input, validation_ground_truths),
        epochs=1
    )
    # Run predictions on the validation data set, and save the experimental run
    validation_network_outputs = model.predict(validation_input)
    save_test(
        event_data_set,
        validation_ground_truths,
        validation_network_outputs,
        epoch=epoch,
        prefix='gravitational_validation_'
    )
    # Run predictions on the part of the training set for which the ground truths are not definitive
    predictions = model.predict(training_input[DEFINITIVE_TRAINING_EXAMPLES:])
    # Convert the predictions to a NumPy array and remove the unnecessary second dimension
    predictions_array = np.array(predictions)[:, 0]
    # Calculate the new ground truths for those examples by adding the gravitational function to the current predictions
    ground_truths = predictions_array + gravitational_ground_truth_offsets(predictions_array, DISTORTION_ROOT, gravity_multiplier)
    training_ground_truths[DEFINITIVE_TRAINING_EXAMPLES:] = ground_truths
    # Expand the dimensions of the new ground truth array so the test saving function will interpret it correctly
    ground_truths_saving = np.expand_dims(ground_truths, axis=1)
    # Save data for comparing the new gravitational ground truths to the original, correct ones
    save_test(
        training_data_set,
        original_training_ground_truths,
        ground_truths_saving,
        epoch=epoch,
        prefix='gravitational_ground_truths_'
    )
    # Add to the gravity multiplier and notify the user of its value
    gravity_multiplier += GRAVITY_MULTIPLIER_INCREMENT
    print('Gravity multiplier is at', gravity_multiplier, 'for epoch', epoch)
