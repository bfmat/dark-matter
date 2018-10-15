#!/usr/bin/env python3
"""A grid search script for the topological convolutional neural network, which calls an independent script that does the training (doing it all in the same place is liable to cause slowdown)"""
# Created by Brendon Matusch, October 2018

import subprocess

from data_processing.deap_topology import create_deap_topology
from models.topological_cnn import TopologicalCNN

# Iterate over each of the possible hyperparameters that can be changed
for num_convolutional_layers in [2, 3, 4, 5]:
    for l2_regularization in [0.0003, 0.001, 0.003, 0.01]:
        for final_convolutional_layer_filters in [2, 4, 8]:
            for kernel_radius_2_first_layer in [0, 1]:
                # Run the script for a single training run
                subprocess.call(['./topological_cnn_grid_search_unit.py', str(num_convolutional_layers), str(l2_regularization), str(final_convolutional_layer_filters), str(kernel_radius_2_first_layer)])
