#!/usr/bin/env python3
"""Training script for a neural network that is trained on a recording in frequency domain split into discrete bands"""
# Created by Brendon Matusch, June 2018

import os

import numpy as np

from data_processing.event_data_set import EventDataSet, RunType
from data_processing.experiment_serialization import save_test
from models.banded_frequency_network import create_model

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
training_input, training_ground_truths, validation_input, validation_ground_truths = event_data_set.banded_frequency_alpha_classification()
# Get the Acoustic Parameter event predictions so they can be compared to the ground truths
training_ap_predictions, validation_ap_predictions = [[event.logarithmic_acoustic_parameter > 0.5 for event in event_list]
                                                      for event_list in [event_data_set.training_events, event_data_set.validation_events]]

# Iterate over the defined number of epochs
for epoch in range(EPOCHS):
    # Train the model on the loaded data set
    model.fit(
        x=training_input,
        y=training_ap_predictions,
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
        prefix='banded_low_resolution_'
    )
    # Save the current model, named with the epoch number
    model_path = os.path.expanduser(f'~/banded_epoch{epoch}.h5')
    model.save(model_path)
