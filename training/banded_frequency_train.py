#!/usr/bin/env python3
"""Training script for a neural network that is trained on a recording in frequency domain split into discrete bands"""
# Created by Brendon Matusch, June 2018

from keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Sequential

from data_processing.event_data_set import EventDataSet, RunType
from data_processing.experiment_serialization import save_test

# Load the event data set from the file, removing multiple-bubble events, disabling acoustic parameter cuts, and keeping background radiation and calibration runs
event_data_set = EventDataSet(
    filter_multiple_bubbles=True,
    filter_acoustic_parameter=False,
    keep_run_types=set([
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
        RunType.CALIFORNIUM_40CM,
        RunType.CALIFORNIUM_60CM,
        RunType.BARIUM_40CM,
        RunType.BARIUM_100CM
    ])
)
# Get the banded frequency domain data and corresponding binary ground truths
training_input, training_ground_truths, validation_input, validation_ground_truths = event_data_set.banded_frequency_alpha_classification()

# The input dimension to the network should be the number of values per banded data point
input_dimension = training_input.shape[1]
# Create a neural network model that includes several dense layers with hyperbolic tangent activations, dropout, and batch normalization
activation = 'tanh'
model = Sequential([
    InputLayer(input_shape=(input_dimension,)),
    BatchNormalization(),
    Dense(24, activation=activation),
    Dropout(0.5),
    Dense(6, activation=activation),
    Dropout(0.5),
    Dense(1)
])
# Output a summary of the model's architecture
print(model.summary())
# Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

# Train the model on the loaded data set
model.fit(
    x=training_input,
    y=training_ground_truths,
    validation_data=(validation_input, validation_ground_truths),
    epochs=20
)
# Run predictions on the validation data set, and save the experimental run
validation_network_outputs = model.predict(validation_input)
save_test(event_data_set, validation_ground_truths, validation_network_outputs)
