#!/usr/bin/env python3
"""A grid search script for the gravitational semi-supervised learning system"""
# Created by Brendon Matusch, July 2018

import copy
import os

import numpy as np

# Use only the second GPU for training (GTX 1060)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from keras.optimizers import SGD

from data_processing.bubble_data_point import RunType
from data_processing.event_data_set import EventDataSet
from data_processing.experiment_serialization import save_test
from models.banded_frequency_network import create_model


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


# Iterate over different values for the number of definitive training examples, the gravity multiplier increment, the distortion root, and the stochastic gradient descent learning rate
for definitive_training_examples in [128, 256]:
    for gravity_multiplier_increment in [0.0005, 0.001, 0.003, 0.005, 0.008]:
        for learning_rate in [0.001, 0.003, 0.01, 0.03]:
            for distortion_root in [3, 5, 7, 9, 11]:
                # Print a few blank lines for separation
                for _ in range(3):
                    print()
                # Document the current hyperparameter combination
                print('HYPERPARAMETERS')
                print('Definitive Training Examples:', definitive_training_examples)
                print('Gravity Multiplier Increment:', gravity_multiplier_increment)
                print('Learning Rate:', learning_rate)
                print('Distortion Root:', distortion_root)
                # Create a description string which is used for saving validation sets
                description = f'gravitational_grid_search_definitive_training_examples{definitive_training_examples}_gravity_multiplier_increment{gravity_multiplier_increment}_learning_rate{learning_rate}_distortion_root{distortion_root}_'

                # Create an instance of the fully connected neural network
                model = create_model()
                # Recompile the model to use a simple stochastic gradient descent optimizer without any momentum or Nesterov; the ground truth tweaking in this system should not be combined with a more complex optimizer, which will interfere with the desired effects
                model.compile(
                    optimizer=SGD(lr=learning_rate),
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
                original_training_ground_truths = training_ground_truths[definitive_training_examples:]
                # Make a copy of the event data set, replacing the validation set with the corresponding training examples for saving the training data
                training_data_set = copy.deepcopy(event_data_set)
                training_data_set.validation_events = \
                    event_data_set.training_events[definitive_training_examples:]
                # Initially, set all of the training ground truths (except for the few for which the original ground truth is kept) to 0.5
                # This keeps the network from learning anything it shouldn't until at least some training has been done on the definitive data
                # It must first be converted to floating-point
                training_ground_truths = training_ground_truths.astype(float)
                training_ground_truths[definitive_training_examples:] = 0.5

                # The gravity multiplier should start at 0 and is added to every epoch
                gravity_multiplier = 0

                # Iterate for a certain number of epochs
                for epoch in range(3000):
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
                        prefix=description + 'validation_'
                    )
                    # Run predictions on the part of the training set for which the ground truths are not definitive
                    predictions = model.predict(training_input[definitive_training_examples:])
                    # Convert the predictions to a NumPy array and remove the unnecessary second dimension
                    predictions_array = np.array(predictions)[:, 0]
                    # Calculate the new ground truths for those examples by adding the gravitational function to the current predictions
                    ground_truths = predictions_array + gravitational_ground_truth_offsets(predictions_array, distortion_root, gravity_multiplier)
                    training_ground_truths[definitive_training_examples:] = ground_truths
                    # Expand the dimensions of the new ground truth array so the test saving function will interpret it correctly
                    ground_truths_saving = np.expand_dims(ground_truths, axis=1)
                    # Save data for comparing the new gravitational ground truths to the original, correct ones
                    save_test(
                        training_data_set,
                        original_training_ground_truths,
                        ground_truths_saving,
                        epoch=epoch,
                        prefix=description + 'ground_truths_'
                    )
                    # Add to the gravity multiplier and notify the user of its value
                    gravity_multiplier += gravity_multiplier_increment
                    print('Gravity multiplier is at', gravity_multiplier, 'for epoch', epoch)
