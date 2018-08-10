#!/usr/bin/env python3
"""Training script that trains the banded frequency neural network on several different hyperparameter combinations"""
# Created by Brendon Matusch, July 2018

from keras.layers import Input, BatchNormalization, Dense, Dropout, concatenate
from keras.models import Model
from keras.regularizers import l2

from data_processing.event_data_set import EventDataSet
from data_processing.bubble_data_point import RunType, load_bubble_frequency_domain
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
# Load training and validation data as NumPy arrays, currying the loading function to disable banding
training_inputs, training_ground_truths, validation_inputs, validation_ground_truths = \
    event_data_set.audio_alpha_classification(
        loading_function=lambda bubble: load_bubble_frequency_domain(bubble, banded=False),
        include_positions=True
    )

# Iterate over the valid configurations for dropout, L2 lambda, and number of hidden layers
for dropout in [0, 0.25, 0.5]:
    for l2_lambda in [0, 0.0003, 0.001, 0.003, 0.01]:
        for hidden_layers in [1, 2, 3]:
            # Print a few blank lines for separation
            for _ in range(3):
                print()
            # Document the current hyperparameter combination
            print('HYPERPARAMETERS')
            print('Dropout:', dropout)
            print('L2 Lambda:', l2_lambda)
            print('Number of Hidden Layers:', hidden_layers)
            # Create a description string containing the hyperparameters, which is used when saving test data
            description = f'high_res_grid_search_dropout{dropout}_l2_lambda{l2_lambda}_hidden_layers{hidden_layers}_'

            # Create a neural network model that includes several dense layers with hyperbolic tangent activations, L2 regularization, and batch normalization
            regularizer = l2(l2_lambda)
            activation = 'tanh'
            # Create two inputs, one for the audio data and one for the position, and concatenate them together
            audio_input = Input((100_002,))
            # Take a separate input for the position, and concatenate it with the audio input
            position_input = Input((3,))
            x = concatenate([audio_input, position_input])
            x = BatchNormalization()(x)
            # If there are 3 hidden layers, include an earlier one with 16 neurons
            if hidden_layers == 3:
                x = Dense(16, activation=activation, kernel_regularizer=regularizer)(x)
                x = Dropout(dropout)(x)
            # Always include a layer with 12 neurons
            x = Dense(12, activation=activation, kernel_regularizer=regularizer)(x)
            x = Dropout(dropout)(x)
            # If there are at least 2 hidden layers, include one with 8 neurons
            if hidden_layers >= 2:
                x = Dense(8, activation=activation, kernel_regularizer=regularizer)(x)
                x = Dropout(dropout)(x)
            output = Dense(1, activation='sigmoid', kernel_regularizer=regularizer)(x)
            # Create a model with both inputs
            model = Model(inputs=[audio_input, position_input], outputs=output)
            # Output a summary of the model's architecture
            print(model.summary())
            # Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['accuracy']
            )

            # Iterate for a certain number of epochs
            for epoch in range(250):
                # Train the model on the loaded data set
                model.fit(
                    x=training_inputs,
                    y=training_ground_truths,
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
                    epoch=epoch,
                    prefix=description
                )
