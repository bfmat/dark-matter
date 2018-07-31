#!/usr/bin/env python3
"""A script used to get relevant metadata of a bubble, provided its unique identifier"""
# Created by Brendon Matusch, July 2018

import sys

from data_processing.event_data_set import EventDataSet
from utilities.verify_arguments import verify_arguments

# A numeric unique bubble identifier is expected
verify_arguments('unique bubble identifier')

# Load that one bubble from the data file
identifier = int(sys.argv[1])
bubble = EventDataSet.load_specific_indices([identifier])[0]

# Print the date, run number, event number, and run type
print('Date:', bubble.date)
print('Run number:', bubble.run_number)
print('Event number:', bubble.event_number)
print('Run type:', bubble.run_type)
