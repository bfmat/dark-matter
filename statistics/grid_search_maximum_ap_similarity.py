#!/usr/bin/env python3
"""Given a grid search AP similarity statistics file, calculate the maximum accuracy (as opposed to mean) for every run"""
# Created by Brendon Matusch, August 2018

import os
import sys

from utilities.verify_arguments import verify_arguments

# A path to a grid search AP similarity statistics file is expected
verify_arguments('path to grid search AP similarity statistics file')
# Load the full contents of the file
with open(os.path.expanduser(sys.argv[1])) as statistics_file:
    statistics_file_contents = statistics_file.read()
# Split it on the separator lines that contain the mean disagreements, removing the last element because it only contains mean data
text_by_run = statistics_file_contents.split('Mean')[:-1]
