#!/usr/bin/env python3
"""Given standard input from the ground truth accuracy calculation script, aggregate the accuracy over the 3 different runs in each configuration"""

import sys

# The number of lines printed by the original script on every run
LINES_PER_RUN = 12
# The number of runs with the same configuration, run in groups
RUNS_PER_CONFIGURATION = 3

# The indices corresponding to the lines containing mean and maximum validation accuracy values within a run
MEAN_ACCURACY_LINE = 9
MAX_ACCURACY_LINE = 10

# Read all lines from standard input
input_lines = sys.stdin.readlines()
# Calculate the total number of runs, based on the known number of lines per run
num_runs = len(input_lines) // LINES_PER_RUN
# Iterate over each of the configurations, based on the number of runs and runs per configuration
for configuration_index in range(num_runs // RUNS_PER_CONFIGURATION):
    # Iterate over the runs within that configuration
    for run_index in range(RUNS_PER_CONFIGURATION):
        # Calculate the starting index of this run
        run_starting_index = ((configuration_index * RUNS_PER_CONFIGURATION) + run_index) * LINES_PER_RUN
        print(run_starting_index)
