#!/usr/bin/env python3
"""Grid search training script for a neural network that is trained on a recording in frequency domain split into discrete bands"""
# Created by Brendon Matusch, October 2018

from keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Sequential
from keras.regularizers import l2

from data_processing.event_data_set import EventDataSet, RunType
from data_processing.experiment_serialization import save_test

# Iterate over possible configurations for the L2 lambda and dropout regularization
for l2_lambda in [0, 0.001, 0.003]:
    for dropout in [0, 0.25, 0.5]:
        # Test each configuration multiple times so the initial set does not cause a bias
        for configuration_test_index in range(3):
            # Create a data set, running fiducial cuts for the most reasonable data
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

            # Print a few blank lines for separation
            for _ in range(3):
                print()
            # Document the current hyperparameter configuration
            print('HYPERPARAMETERS')
            print('L2 lambda:', l2_lambda)
            print('Dropout:', dropout)
            print('Configuration test index:', configuration_test_index)
            # Create a description folder path that is used when saving validation data
            description = f'banded_all/dropout{dropout}/l2_lambda{l2_lambda}/configuration_test{configuration_test_index}'

            # Create a neural network model that includes several dense layers with hyperbolic tangent activations, L2 regularization, and batch normalization
            regularizer = l2(l2_lambda)
            activation = 'tanh'
            model = Sequential([
                InputLayer(input_shape=(16,)),
                BatchNormalization(),
                Dense(12, activation=activation, kernel_regularizer=regularizer),
                Dropout(dropout),
                Dense(8, activation=activation, kernel_regularizer=regularizer),
                Dropout(dropout),
                Dense(1, activation='sigmoid', kernel_regularizer=regularizer)
            ])
            # Output a summary of the model's architecture
            print(model.summary())
            # Use a mean squared error loss function and an Adam optimizer, and print the accuracy while training
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['accuracy']
            )

            # Iterate over a certain number of epochs
            for epoch in range(6000):
                # Train the model on the loaded data set
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
                    prefix=description
                )
