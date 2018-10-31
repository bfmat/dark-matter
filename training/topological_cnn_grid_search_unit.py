#!/usr/bin/env python3
"""A training script for a single run of topological convolutional neural network within a grid search (called to prevent memory leaks and slowdown)"""
# Created by Brendon Matusch, October 2018

import sys

from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2

from data_processing.deap_topology import create_deap_topology
from models.topological_cnn import TopologicalCNN

# Load the hyperparameters for this unit from the command line arguments
num_convolutional_layers = int(sys.argv[1])
l2_regularization = float(sys.argv[2])
final_convolutional_layer_filters = int(sys.argv[3])
kernel_radius_2_first_layer = int(sys.argv[4])
nuclear_recoil_weight = float(sys.argv[5])
activation = sys.argv[6]
configuration_test = int(sys.argv[7])
# Load the DEAP data as a topology
topology = create_deap_topology()
# Create an L2 regularizer to use for all layers
regularizer = l2(l2_regularization)
# Create a list of convolutional layers to add to
convolutional_layers = []
# Iterate over the number of layers to add
for layer_index in range(num_convolutional_layers):
    # Subtract the index from one less than the number of layers to get the distance from the end
    distance_from_last_layer = (num_convolutional_layers - 1) - layer_index
    # The number of filters should be increased by 50% every layer from the end, to the nearest integer
    filters = int(round(final_convolutional_layer_filters * (1.5 ** distance_from_last_layer)))
    # If it is enabled and this is the first layer, set the kernel radius to 2; otherwise make it 1
    kernel_radius = 2 if (kernel_radius_2_first_layer and layer_index == 0) else 1
    # Create such a layer and add it to the list
    convolutional_layers.append({'kernel_radius': kernel_radius, 'filters': filters, 'activation': activation, 'regularizer': regularizer})
# Print a few blank lines for separation
for _ in range(3):
    print()
# Document the current hyperparameter combination
print('HYPERPARAMETERS')
print('Convolutional Layers:', num_convolutional_layers)
print('L2 Lambda:', l2_regularization)
print('Final Convolutional Layer Filters:', final_convolutional_layer_filters)
print('Kernel Radius of 2 on First Layer:', kernel_radius_2_first_layer)
print('Nuclear Recoil Class Weight:', nuclear_recoil_weight)
print('Activation:', activation)
print('Configuration Test:', configuration_test)
# Train a neural network with this list of convolutional layers
TopologicalCNN(
    surface_topology_set=topology,
    convolutional_layers=convolutional_layers,
    remaining_model=Sequential([Dense(1, activation='sigmoid')]),
    optimizer='adam',
    loss='mse',
    epochs=50,
    validation_size=500,
    class_weight={0: nuclear_recoil_weight, 1: 1.0}
)
