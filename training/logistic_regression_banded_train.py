#!/usr/bin/env python3
"""Training script for a linear regression that is trained on a recording in frequency domain split into discrete bands"""
# Created by Brendon Matusch, October 2018

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from data_processing.event_data_set import EventDataSet, RunType
from data_processing.experiment_serialization import save_test

# Load the event data set from the file, removing multiple-bubble events, disabling acoustic parameter cuts, and keeping background radiation and calibration runs
event_data_set = EventDataSet(
    keep_run_types={
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
        RunType.CALIFORNIUM
    },
    use_wall_cuts=True
)
# Get the banded frequency domain data and corresponding binary ground truths
training_input, training_ground_truths, validation_input, validation_ground_truths = event_data_set.banded_frequency_alpha_classification()
# Add polynomial features up to degree 3 to the input matrices
polynomial_features = PolynomialFeatures(degree=3)
training_input = polynomial_features.fit_transform(training_input)
validation_input = polynomial_features.fit_transform(validation_input)

# Create a logistic regression model and train it on the banded frequency data
logistic_regression = LogisticRegression()
logistic_regression.fit(training_input, training_ground_truths)
# Print accuracy on the training and validation sets
print('Training accuracy:', logistic_regression.score(training_input, training_ground_truths))
print('Validation accuracy:', logistic_regression.score(validation_input, validation_ground_truths))
# Run floating-point predictions on the validation set
validation_predictions = logistic_regression.decision_function(validation_input)
# Expand the array, adding a second axis because it's expected by the saving function
validation_predictions = np.expand_dims(validation_predictions, axis=1)
# Save the resulting validation set to disk
save_test(
    event_data_set,
    validation_ground_truths,
    validation_predictions,
    prefix='banded_logistic_'
)
