#!/usr/bin/env python3
"""A script for loading an example topology and plotting the nodes and the connections between them"""
# Created by Brendon Matusch, August 2018

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from data_processing.surface_topology import SurfaceTopologySet

# Load the example JSON data set
topology = SurfaceTopologySet('../data_processing/example_surface_topology_set.json')
# Create a list of lines to add to, and another list for corresponding colors
lines = []
colors = []
# Iterate over each of the nodes, creating lines to connected nodes
for node in topology.nodes:
    # Iterate over the connected nodes, alongside numbers that represent the index of the connection
    for connection_index, connected_node in enumerate(node.connected_nodes):
        # If the connected node is absent, skip to the next iteration
        if connected_node is None:
            continue
        # Otherwise, get the delta vector from the main node to the connected node
        main_node_position = np.array([node.x_position, node.y_position])
        connected_node_position = np.array([connected_node.x_position, connected_node.y_position])
        delta = connected_node_position - main_node_position
        # Calculate the length of the delta vector and normalize it so the length is 1
        delta_length = np.linalg.norm(delta)
        # Divide the delta by its length times a certain value to normalize it to a specific length
        normalized_delta = delta / (delta_length * 6)
        # Add a line showing the normalized delta, away from the main node
        lines.append([main_node_position, main_node_position + normalized_delta])
        # Add a color corresponding to the index of the connection
        colors.append({0: 'r', 1: 'g', 2: 'b'}[connection_index])

# Get the axes in order to display lines
axes = plt.subplot()
# Display the vector lines in the defined colors
axes.add_collection(LineCollection(segments=lines, colors=colors))
# Scatter plot the X and Y positions of all of the nodes (ignoring Z for now)
plt.scatter(
    x=[node.x_position for node in topology.nodes],
    y=[node.y_position for node in topology.nodes]
)
# Display the graph on screen
plt.show()
