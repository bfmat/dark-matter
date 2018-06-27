#!/usr/bin/python3
"""A script for converting ROOT data to a smaller, quicker to load, and more portable Pickle format"""
# Created by Brendon Matusch, June 2018

import os
import pickle

from bubble_data_point import BubbleDataPoint
from event_data import EVENT_FILE_PATH

import ROOT

# Open the event file and get the main tree
# These cannot be in the same line or a segmentation fault will occur
event_file = ROOT.TFile('/opt/merged_all_all.root')
tree = event_file.Get('T')
# Iterate over the tree with a corresponding index, and convert the bubbles to a custom data class
bubbles = [BubbleDataPoint(event, index) for index, event in enumerate(tree)]
# Serialize the list to a Pickle binary file and notify the user
with open(EVENT_FILE_PATH, 'wb') as pickle_file:
    pickle.dump(bubbles, pickle_file)
print('Saved as', EVENT_FILE_PATH)
