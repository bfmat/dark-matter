#!/usr/bin/env python3
"""A script for loading an example topology and plotting the nodes and the connections between them"""
# Created by Brendon Matusch, August 2018

import matplotlib.pyplot as plt

from data_processing.surface_topology import SurfaceTopologySet

# Load the example JSON data set
topology = SurfaceTopologySet('example_surface_topology_set.json')
# Scatter plot the X and Y positions of all of the nodes (ignoring Z for now)
plt.scatter(
    x=[node.x_position for node in topology.nodes],
    y=[node.y_position for node in topology.nodes]
)
# Display the graph on screen
plt.show()
