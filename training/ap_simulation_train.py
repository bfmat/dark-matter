#!/usr/bin/env python3
"""Training script for a neural network that is trained on banded frequency domain information to simulate the Acoustic Parameter function"""
# Created by Brendon Matusch, June 2018

import os

from data_processing.event_data_set import EventDataSet, RunType
from data_processing.experiment_serialization import save_test
from models.ap_simulation_network import create_model

# The number of epochs to train for
EPOCHS = 250

# Create an instance of the fully connected neural network
model = create_model()

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
training_input, training_ground_truths, validation_input, validation_ground_truths = event_data_set.ap_simulation()

# Iterate over the defined number of epochs
for epoch in range(EPOCHS):
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
        prefix='ap_simulation_'
    )
    # # Save the current model, named with the epoch number
    # model_path = os.path.expanduser(f'~/ap_simulation{epoch}.h5')
    # model.save(model_path)
