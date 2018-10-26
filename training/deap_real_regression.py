#!/usr/bin/env python3
"""Train a logistic regression model on the small number of real-world data points"""
# Created by Brendon Matusch, October 2018

import numpy as np
from sklearn.linear_model import LogisticRegression

from data_processing.load_deap_data import load_real_world_deap_data
from training.pulse_count_train import prepare_events, evaluate_predictions

# Load all real-world test events from the file
real_world_neck_events, real_world_neutron_events = load_real_world_deap_data()
# Prepare the input and ground truth data for training
inputs, ground_truths, events = prepare_events(real_world_neck_events, real_world_neutron_events)
# Separate the lists into training and validation sets
(training_inputs, validation_inputs), (training_ground_truths, validation_ground_truths), (training_events, validation_events) = ((array[:20], array[20:]) for array in [inputs, ground_truths, events])
# Create a logistic regression classifier
classifier = LogisticRegression()
# Fit the classifier on the training input and output arrays
classifier.fit(training_inputs, training_ground_truths)
# Score the classifier on the training and validation arrays separately
training_score = classifier.score(training_inputs, training_ground_truths)
validation_score = classifier.score(validation_inputs, validation_ground_truths)
print('Training score:', training_score)
print('Validation score:', validation_score)
# Save the model's predictions on the training and validation sets
training_predictions = classifier.decision_function(training_inputs)
validation_predictions = classifier.decision_function(validation_inputs)
evaluate_predictions(training_ground_truths, training_predictions, training_events, 0, 'real_regression_training')
evaluate_predictions(validation_ground_truths, validation_predictions, validation_events, 0, 'real_regression_validation')
