#!/usr/bin/env python3
"""A training script for the topological convolutional neural network on the DEAP pulse count data"""
# Created by Brendon Matusch, August 2018

from keras.layers import Dense
from keras.models import Sequential

from data_processing.deap_topology import create_deap_topology
from models.topological_cnn import TopologicalCNN

# Load the DEAP data as a topology
topology = create_deap_topology()
# Train a network with 1 convolutional layer and 1 dense layer on it (discarding the trained model)
TopologicalCNN(
    surface_topology_set=topology,
    convolutional_layers=[
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'},
        {'kernel_radius': 1, 'filters': 16, 'activation': 'tanh'}
    ],
    remaining_model=Sequential([Dense(1, activation='sigmoid')]),
    optimizer='adam',
    loss='mse',
    epochs=500
)
