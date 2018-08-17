"""Code related to a data structure containing the triangulated surface of a 2-dimensional object, with a set of numeric neural network input values corresponding to each triangle"""
# Created by Brendon Matusch, August 2018

import json
import os

import numpy as np


class SurfaceTopologyNode:
    """A single topology node, which contains information loaded from one dictionary in the JSON file"""

    # A placeholder for references to the nodes above, right and left (triangular directions), in anticlockwise order, starting with any of the 3; any 0, 1 or 2 of these may be undefined
    connected_nodes = None

    # A placeholder for the index, in the list of references to the nodes, to the vertically highest connected node
    highest_node_index = None

    def __init__(self, json_dictionary):
        """Load the relevant components from a JSON dictionary corresponding to a node"""
        # Get the unique identifier that defines this node
        self.identifier = json_dictionary['id']
        # Get the 3 values in the position individually (Y is vertical)
        self.x_position, self.y_position, self.z_position = json_dictionary['position']
        # Take the raw list of connections for now; up, right, and left will be calculated later
        # They are defined to be in anticlockwise order, starting with any of the 3
        self.raw_connections_anticlockwise = json_dictionary['connections']
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
            node.connected_nodes = [
                # Get the node corresponding to the connected identifier
                self.get_node(node_identifier)
                # Let None pass through; don't try to load anything in that case
                if node_identifier is not None else None
                # The identifiers are in the original order
                for node_identifier in node.raw_connections_anticlockwise
            ]
            # Get the Y (vertical) positions of these connected nodes, replacing None with negative infinity (so empty connections will never be considered the farthest up)
            y_positions = [connected_node.y_position if connected_node is not None else float('-inf') for connected_node in node.connected_nodes]
            # Find the index of the node with the greatest vertical position
            # That may be below this node if it is at the top of a convex surface
            # This specific edge case will have to be handled when calculating kernels
            node.highest_node_index = np.argmax(y_positions)

    def get_node(self, identifier: int) -> SurfaceTopologyNode:
        """Return a reference to a node given its identifier"""
        # Search for the node with this ID rather than just using the index; they may be out of order
        # There should only be one, so taking the first element is fine
        return [node for node in self.nodes if node.identifier == identifier][0]
