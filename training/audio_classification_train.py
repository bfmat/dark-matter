#!/usr/bin/env python3
"""Training script for a neural network that classifies audio samples into alpha particles and neutrons"""
# Created by Brendon Matusch, June 2018

import os

from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Activation, Flatten, Dropout, InputLayer, BatchNormalization, Reshape
from keras.models import Sequential

from data_processing.event_data_set import EventDataSet
from data_processing.bubble_data_point import RunType, load_bubble_audio
from data_processing.experiment_serialization import save_test

# Create a one-dimensional convolutional neural network model with hyperbolic tangent activations
# It should take both microphone channels and an entire clip of audio
activation = 'tanh'
model = Sequential()
model.add(InputLayer(input_shape=(250000, 2)))
model.add(BatchNormalization())
model.add(Conv1D(filters=48, kernel_size=80,
                 strides=4, activation=activation))
model.add(MaxPooling1D(6))
model.add(BatchNormalization())
for _ in range(3):
    model.add(Conv1D(filters=48, kernel_size=3))
    model.add(BatchNormalization())
model.add(MaxPooling1D(6))
model.add(BatchNormalization())
for _ in range(4):
    model.add(Conv1D(filters=96, kernel_size=3))
    model.add(BatchNormalization())
model.add(MaxPooling1D(6))
model.add(BatchNormalization())
for _ in range(6):
    model.add(Conv1D(filters=192, kernel_size=3))
    model.add(BatchNormalization())
model.add(MaxPooling1D(6))
model.add(BatchNormalization())
for _ in range(3):
    model.add(Conv1D(filters=384, kernel_size=3))
    model.add(BatchNormalization())
model.add(Reshape((-1, 1)))
model.add(GlobalAveragePooling1D())
model.add(Activation('sigmoid'))

# Output a summary of the model's architecture
print(model.summary())
# Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

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
# Create a training data generator with the audio loading function
training_generator_callable, validation_inputs, validation_ground_truths = event_data_set.arbitrary_alpha_classification_generator(
    data_converter=load_bubble_audio,
    storage_size=512,
    batch_size=32,
    examples_replaced_per_batch=16
)
training_generator = training_generator_callable()

# Iterate over training and validation for 20 epochs
for epoch in range(20):
    # Train the model on the generator
    model.fit_generator(
        training_generator,
        steps_per_epoch=128,
        epochs=1
    )
    # Evaluate the model on the validation data set
    loss, accuracy = model.evaluate(
        x=validation_inputs,
        y=validation_ground_truths,
        verbose=0
    )
    # Output the validation loss and accuracy to the user
    print('Validation loss:', loss)
    print('Validation accuracy:', accuracy)
    # Run predictions on the validation data set, and save the experimental run
    validation_network_outputs = model.predict(validation_inputs)
    save_test(
        event_data_set,
        validation_ground_truths,
        validation_network_outputs,
        epoch
    )
    # Save the current model, named with the epoch number
    model_path = os.path.expanduser(f'~/epoch{epoch}.h5')
    model.save(model_path)
