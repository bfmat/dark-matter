#!/usr/bin/env python3
"""A script for printing out statistics about the nested list output by one of the two parsing scripts"""
# Created by Brendon Matusch, November 2018

import json
import sys

# Parse the data list from standard input as JSON
data = json.load(sys.stdin)
# Calculate the minimum false positive percentage for runs with 100% recall
# Recall refers to the number of alphas identified as such; false positive percentage refers to the number of recoils mis-identified as alphas
min_false_perfect_recall = min([false_positive for (recall, false_positive) in data if recall == 1])
print('Minimum false positives of runs with perfect recall:', min_false_perfect_recall)
# Repeat this for runs with at least 97% recall
min_false_97_recall = min([false_positive for (recall, false_positive) in data if recall >= 0.97])
print('Minimum false positives of runs with at least 97% recall:', min_false_97_recall)

# Print the maximum recall of runs with at most 87% false positives
max_recall_87_false = max([recall for (recall, false_positive) in data if false_positive <= 0.87])
print('Maximum recall of runs with at most 87% false positives:', max_recall_87_false)
# Repeat this for runs with at most 70% false positives
max_recall_70_false = max([recall for (recall, false_positive) in data if false_positive <= 0.7])
print('Maximum recall of runs with at most 70% false positives:', max_recall_70_false)
