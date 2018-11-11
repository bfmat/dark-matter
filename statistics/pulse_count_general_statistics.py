#!/usr/bin/env python3
"""A script for printing out statistics about the nested list output by one of the two parsing scripts"""
# Created by Brendon Matusch, November 2018

import json
import sys

# Parse the data list from standard input as JSON
data = json.load(sys.stdin)
# Recall refers to the number of alphas identified as such; false positive percentage refers to the number of recoils mis-identified as alphas
# Calculate the minimum false positive percentage for runs with at least 99.6% recall
min_false_99_6_recall = min([false_positive for (recall, false_positive) in data if recall >= 0.996])
print('Minimum false positives of runs with at least 99.6% recall:', min_false_99_6_recall)
# Print the maximum recall of runs with at most 91% false positives
max_recall_91_false = max([recall for (recall, false_positive) in data if false_positive <= 0.91])
print('Maximum recall of runs with at most 91% false positives:', max_recall_91_false)
