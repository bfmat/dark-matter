#!/usr/bin/env python3
"""A script for printing out statistics about the nested list output by one of the two parsing scripts"""
# Created by Brendon Matusch, November 2018

import json
import sys

import numpy as np

# Parse the data list from standard input as JSON and convert it to a NumPy array
data = np.array(json.load(sys.stdin))
print(data)
