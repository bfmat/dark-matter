#!/usr/bin/env python3
"""A script for running inference with a trained classification model on a data generator, and saving its outputs and the ground truths on the validation set and a portion of the training set"""
# Created by Brendon Matusch, June 2018

import os
import sys

import numpy as np
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape, Activation, Flatten, Dense, Dropout, InputLayer, BatchNormalization
from keras.models import Sequential, load_model

from data_processing.event_data_set import EventDataSet
from data_processing.bubble_data_point import RunType, load_bubble_audio, load_bubble_images
from data_processing.experiment_serialization import save_test
from utilities.verify_arguments import verify_arguments

# The number of bubbles to load from the training data set for evaluation
TRAINING_SET_BUBBLES = 4

# This script should take a keyword identifying the data set to test on, and a trained model
verify_arguments('"waveform" or "image"', 'trained model')

# Load the event data set from the file, removing multiple-bubble events, disabling acoustic parameter cuts, and keeping background radiation and calibration runs
event_data_set = EventDataSet(
    filter_multiple_bubbles=True,
    keep_run_types=set([
        RunType.LOW_BACKGROUND,
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
        RunType.AMERICIUM_BERYLLIUM,
        RunType.CALIFORNIUM_40CM,
        RunType.CALIFORNIUM_60CM,
        RunType.BARIUM_100CM,
        RunType.BARIUM_40CM
    ]),
    use_fiducial_cuts=False
)
# Choose a data loading function that depends on the keyword in the first argument
keyword = sys.argv[1].strip().lower()
loading_function = load_bubble_audio if keyword == 'waveform' else load_bubble_images
# Run the function to create a generator but ignore the result; we will read the training and validation bubble lists separately
event_data_set.arbitrary_alpha_classification_generator(
    data_converter=loading_function,
    storage_size=512,
    batch_size=32,
    examples_replaced_per_batch=16
)
# Combine the validation set and the beginning of the training set into a single list, and iterate over it, adding the data to lists for inputs and ground truths
inputs = []
ground_truths = []
for bubble in event_data_set.validation_events + event_data_set.training_events[:TRAINING_SET_BUBBLES]:
    # Run the bubble through the loading function to get the input
    inputs.append(loading_function(bubble))
    # Predict the bubble is an alpha particle if it is in the background radiation set
    ground_truths.append(bubble.run_type == RunType.LOW_BACKGROUND)
# Convert the input and ground truth lists to NumPy arrays
inputs_array = np.concatenate(inputs)
ground_truths_array = np.array(ground_truths)

# Create a one-dimensional convolutional neural network model with rectified linear activations
# It should take both microphone channels and an entire clip of audio
activation = 'relu'
model = Sequential()
model.add(InputLayer(input_shape=(250000, 2)))
model.add(BatchNormalization())
model.add(Conv1D(filters=48, kernel_size=80,
                 strides=4, activation=activation))
model.add(MaxPooling1D(6))
model.add(BatchNormalization())
for _ in range(3):
    model.add(Conv1D(filters=48, kernel_size=3, activation=activation))
    model.add(BatchNormalization())
model.add(MaxPooling1D(6))
model.add(BatchNormalization())
for _ in range(4):
    model.add(Conv1D(filters=96, kernel_size=3, activation=activation))
    model.add(BatchNormalization())
model.add(MaxPooling1D(6))
model.add(BatchNormalization())
for _ in range(6):
    model.add(Conv1D(filters=192, kernel_size=3, activation=activation))
    model.add(BatchNormalization())
model.add(MaxPooling1D(6))
model.add(BatchNormalization())
for _ in range(3):
    model.add(Conv1D(filters=384, kernel_size=3, activation=activation))
    model.add(BatchNormalization())
model.add(Reshape((-1, 1)))
model.add(GlobalAveragePooling1D())
model.add(Activation('sigmoid'))

# Load the trained weights from disk
model.load_weights(os.path.expanduser(sys.argv[2]))
# Run inference on the combined training and validation inputs
network_outputs = model.predict(inputs_array)
# Save this experiment so it can be graphed
save_test(event_data_set, ground_truths_array, network_outputs)
