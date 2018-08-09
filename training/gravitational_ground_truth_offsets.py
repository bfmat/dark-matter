#!/usr/bin/env python3
"""A function used to distort a neural network's predictions and produce ground truth offsets that push unlabeled examples further to the side they are already nearest"""
# Created by Brendon Matusch, July 2018

import numpy as np


def gravitational_ground_truth_offsets(predictions: np.ndarray, distortion_power: float, gravity_multiplier: float) -> np.ndarray:
    """Get an array of ground truth offsets based on a gravitational model where examples classified very close to one edge will be pulled toward that edge, and examples near the middle will make little difference"""
    # The function should pass through (0.5, 0), should change very little near that point, and should rapidly asymptote in the negative or positive directions as the prediction comes close to 0 or 1
    # First, scale the predictions to the range of -1 to 1
    predictions_scaled = (predictions - 0.5) * 2
    # Take the hyperbolic tangent so examples in the middle are affected minimally
    hyperbolic_tangent = np.tanh(predictions_scaled)
    # Take the Nth power (removing the sign and multiplying it back in after so the negative side is the same as the positive side) of the hyperbolic tangent so that the area around 0 is squashed
    power_distorted = np.sign(hyperbolic_tangent) * np.power(np.abs(hyperbolic_tangent), distortion_power)
    # Multiply it by a constant so the gravitational offset does not dominate the training process
    return power_distorted * gravity_multiplier
