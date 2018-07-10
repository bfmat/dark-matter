#!/usr/bin/env python3
"""A script for extracting various statistics related to the distribution of classes in the data set"""
# Created by Brendon Matusch, July 2018

from data_processing.bubble_data_point import RunType
from data_processing.event_data_set import EventDataSet

# Load the data set without running any cuts other than the basic garbage filters
all_run_types = [
    RunType.LOW_BACKGROUND,
    RunType.AMERICIUM_BERYLLIUM,
    RunType.CALIFORNIUM_40CM,
    RunType.CALIFORNIUM_60CM,
    RunType.BARIUM_100CM,
    RunType.BARIUM_40CM
]
event_data_set = EventDataSet(
    filter_multiple_bubbles=False,
    keep_run_types=set(all_run_types),
    use_fiducial_cuts=False
)
# Combine the training and validation lists
bubbles = event_data_set.training_events + event_data_set.validation_events
# Iterate over ech of the run types
for run_type in all_run_types:
    # Print the overall count of each run type
    run_type_bubbles = [bubble for bubble in bubbles
                        if bubble.run_type == run_type]
    print(len(run_type_bubbles), 'bubbles that are instances of', run_type.name)
    # Print the number that qualify as single bubbles
    single_bubble_instances = len([
        bubble for bubble in run_type_bubbles
        if bubble.num_bubbles_image == 1
        and bubble.num_bubbles_pressure >= 0.8 and bubble.num_bubbles_pressure <= 1.2
    ])
    print(single_bubble_instances, 'single bubble events')
    # Print the number that pass the fiducial cuts
    pass_fiducial_cuts = len([
        bubble for bubble in run_type_bubbles
        if EventDataSet.is_not_wall_event(bubble)
    ])
    print(pass_fiducial_cuts, 'that pass fiducial cuts')
    # Print the number for that are single bubbles and also pass the fiducial cuts
    pass_both_tests = len([
        bubble for bubble in run_type_bubbles
        if bubble.num_bubbles_image == 1
        and bubble.num_bubbles_pressure >= 0.8 and bubble.num_bubbles_pressure <= 1.2
        and EventDataSet.is_not_wall_event(bubble)
    ])
    print(pass_both_tests, 'pass both tests')
    # Print a blank line for separation
    print()
