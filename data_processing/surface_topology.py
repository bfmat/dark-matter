"""Code related to a data structure containing the triangulated surface of a 2-dimensional object, with a set of numeric neural network input values corresponding to each triangle"""
# Created by Brendon Matusch, August 2018

import json
import os

import numpy as np


class SurfaceTopologyNode:
    """A single topology node, which contains information loaded from one dictionary in the JSON file"""

    # A placeholder for references to the nodes above, right and left (triangular directions), in clockwise order
    # The first node (above) is guaranteed to be defined (even if it is physically below this node); the other 2 are defined if possible
    connected_nodes = []

    def __init__(self, json_dictionary):
        """Load the relevant components from a JSON dictionary corresponding to a node"""
        # Get the unique identifier that defines this node
        self.identifier = json_dictionary['id']
        # Get the 3 values in the position individually (Y is vertical)
        self.x_position, self.y_position, self.z_position = json_dictionary['position']
        # Take the raw list of connections for now; up, right, and left will be calculated later
        # They are defined to be in clockwise order, starting with any of the 3
        self.raw_connections_clockwise = json_dictionary['connections']
        # Take the list of values directly; the topology set class will handle processing when the data is loaded
        self.values = json_dictionary['values']


class SurfaceTopologySet:
    """A data structure containing the triangulated surface of a 2-dimensional object, with a set of numeric neural network input values corresponding to each triangle"""

    def __init__(self, json_path: str):
        """Create a surface topology set, given a path to a JSON file containing the relevant data"""
        # Open the JSON file and load all data out of it
        with open(os.path.expanduser(json_path)) as json_file:
            json_data = json.load(json_file)
        # Load a node object corresponding to each JSON dictionary
        self.nodes = [SurfaceTopologyNode(json_dictionary) for json_dictionary in json_data]
        # Iterate over the nodes, calculating their relationships to other nodes
        for node in self.nodes:
            # Get the nodes this one is connected to
            connected_nodes_clockwise = [
                # Find the node with this ID rather than just using the index; they may be out of order
                [other_node for other_node in self.nodes if other_node.identifier == node_identifier][0]
                # Let None pass through; don't try to load anything in that case
                if node_identifier is not None else None
                # The identifiers are in the original order
                for node_identifier in node.raw_connections_clockwise
            ]
            # Get the Y (vertical) positions of these connected nodes, replacing None with negative infinity (so empty connections will never be considered the farthest up)
            y_positions = [connected_node.y_position if connected_node is not None else float('-inf') for connected_node in connected_nodes_clockwise]
            # Get the index of the node with the greatest vertical position
            # That may be below this node if it is at the top of a convex surface
            # This specific edge case will have to be handled when calculating kernels
            highest_node_index = np.argmax(y_positions)
            # Define the properly ordered list of connections (where the first is vertically the highest) by offsetting each of the node indices
            # The goal is to rotate the list of connections to the top node comes first
            node.connected_nodes = [connected_nodes_clockwise[(highest_node_index + index_offset) % len(connected_nodes_clockwise)] for index_offset in range(len(connected_nodes_clockwise))]
