#!/usr/bin/env python3
"""A training script for a neural network that predicts the position of a bubble based on the times at which the audio signal arrives at various piezos"""
# Created by Brendon Matusch, July 2018

from data_processing.event_data_set import EventDataSet, RunType
from data_processing.experiment_serialization import save_test

# Load the event data set from the file, removing multiple-bubble events, disabling acoustic parameter cuts, and keeping background radiation and calibration runs
event_data_set = EventDataSet(
    keep_run_types={
        RunType.LOW_BACKGROUND,
        RunType.AMERICIUM_BERYLLIUM,
        RunType.CALIFORNIUM
    },
    use_wall_cuts=False
)

# Get the zero time arrays and corresponding position ground truths
training_input, training_ground_truths, validation_input, validation_ground_truths = event_data_set.position_from_time_zero()
