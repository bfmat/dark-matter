#!/usr/bin/env python3
"""A script for extracting various statistics related to the distribution of classes in the data set"""
# Created by Brendon Matusch, July 2018

import random

from data_processing.bubble_data_point import RunType
from data_processing.event_data_set import EventDataSet

# Load all bubble events straight from disk
bubbles = EventDataSet.load_data_from_file()
# Iterate over all of the run types, excepting garbage
all_run_types = [
    RunType.LOW_BACKGROUND,
    RunType.AMERICIUM_BERYLLIUM,
    RunType.CALIFORNIUM,
    RunType.BARIUM,
    RunType.COBALT
]
for run_type in all_run_types:
    # Print the overall count of each run type
    run_type_bubbles = [bubble for bubble in bubbles
                        if bubble.run_type == run_type]
    print(len(run_type_bubbles), 'bubbles that are instances of', run_type.name)
    # Print the number that pass standard cuts for quality and number of bubbles
    passing_standard_cuts = [
        bubble for bubble in run_type_bubbles
        if EventDataSet.passes_standard_cuts(bubble)
    ]
    print(len(passing_standard_cuts), 'that pass standard cuts')
    # Print the number that also pass the validation cuts
    passing_validation_cuts = [
        bubble for bubble in passing_standard_cuts
        if EventDataSet.passes_validation_cuts(bubble)
    ]
    print(len(passing_validation_cuts), 'that also pass validation cuts')
    # Shuffle the list of bubbles that pass the validation cuts and randomly select a few to output
    random.shuffle(passing_validation_cuts)
    examples = passing_validation_cuts[:10]
    # Print the date, run number, and event number for those chosen examples
    print('Examples that pass standard and validation cuts:')
    for example in examples:
        print(f'{example.date}, run {example.run_number}, event {example.event_number}')
    # Print a blank line for separation
    print()
