#!/usr/bin/env python3
"""A grid search script for the topological convolutional neural network, which calls an independent script that does the training (doing it all in the same place is liable to cause slowdown)"""
# Created by Brendon Matusch, October 2018

import subprocess

# Iterate over each of the possible hyperparameters that can be changed
for num_convolutional_layers in [3, 6]:
    for l2_regularization in [0, 0.0005, 0.001, 0.003]:
        for final_convolutional_layer_filters in [8]:
            for kernel_radius_2_first_layer in [0]:
                for nuclear_recoil_weight in [0.005, 0.0075, 0.01, 0.015, 0.02]:
                    # Test each configuration multiple times to verify its performance
                    for configuration_test in range(6):
                        # Run the script for a single training run
                        subprocess.call(['./topological_cnn_grid_search_unit.py', str(num_convolutional_layers), str(l2_regularization),
                                         str(final_convolutional_layer_filters), str(kernel_radius_2_first_layer), str(nuclear_recoil_weight), str(configuration_test)])
