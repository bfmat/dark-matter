#!/usr/bin/env python3
"""A minimal example training script for the topological convolutional neural network"""
# Created by Brendon Matusch, August 2018

from keras.layers import Dense
from keras.models import Sequential

from data_processing.surface_topology import SurfaceTopologySet
from models.topological_cnn import TopologicalCNN

# Load the example JSON data set
topology = SurfaceTopologySet('../data_processing/example_surface_topology_set.json')
# Train a network with 2 convolutional layers and 1 dense layer on it (discarding the trained model)
TopologicalCNN(
    surface_topology_set=topology,
    convolutional_layers=[
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'},
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'},
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'},
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'},
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'},
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'},
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'},
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'},
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'},
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'},
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'}
    ],
    remaining_model=Sequential([Dense(1)]),
    optimizer='adam',
    loss='mse',
    epochs=1000
)
