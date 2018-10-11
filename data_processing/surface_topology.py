"""Code related to a data structure containing the triangulated surface of a 2-dimensional object, with a set of numeric neural network input values corresponding to each triangle"""
# Created by Brendon Matusch, August 2018

import csv

from typing import List, Optional, Tuple

import numpy as np


class SurfaceTopologyNode:
    """A single topology node, which contains information loaded from one line in the CSV file"""

    # A placeholder for references to the connected nodes, in clockwise order, possibly including -1 if there are missing connections
    connected_nodes = None

    def __init__(self, identifier, position, connections, values) -> None:
        """Set the relevant components that are provided"""
        # Get the unique identifier that defines this node
        self.identifier = identifier
        # Get the 3 values in the position individually (Y is vertical)
        self.x_position, self.y_position, self.z_position = position
        # Take the raw list of connections for now; up, right, and left will be calculated later
        # They are defined to be in clockwise order, starting with any of them
        self.raw_connections_clockwise = connections
        # Take the list of values directly; this class is not responsible for any processing
        self.values = values


class SurfaceTopologySet:
    """A data structure containing the triangulated surface of a 2-dimensional object, with a set of numeric neural network input values corresponding to each triangle"""

    # The list of nodes belonging to this set, starting as an empty list
    nodes = []

    def __init__(self, csv_path: str, values: List[List[float]], positions: List[Tuple[float]], ground_truths: List[bool]) -> None:
        """Create a surface topology set, given a path to a CSV file containing the relevant data, a nested list of values for each of the nodes, a 3D position for each of the nodes, and the corresponding ground truths"""
        # Open the CSV file containing all of the connections between nodes
        with open(csv_path) as connection_file:
            # Create a CSV reader to load the file
            connection_reader = csv.reader(connection_file)
            # Iterate over the lines of the file, which contain 7 string values in the form (node, connected node 0..., connected node 5), alongside corresponding positions and lists of values
            for line, position, node_values in zip(connection_reader, positions, values):
                # Convert all of the strings to integers
                line_numeric = [int(string) for string in line]
                # The first element of the line is the identifier
                identifier = line_numeric[0]
                # The remainder is the list of connections
                connections = line_numeric[1:]
                # Create a node object using these values
                node = SurfaceTopologyNode(identifier, position, connections, node_values)
                # Add it to the global list of nodes
                self.nodes.append(node)
        # Iterate over the nodes, calculating their relationships to other nodes
        for node in self.nodes:
            # Get the nodes this one is connected to
            connected_nodes_clockwise = [
                # Get the node corresponding to the connected identifier
                self.get_node(node_identifier)
                # Let -1 pass through; don't try to load anything in that case
                if node_identifier != -1 else None
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
            node.connected_nodes = [connected_nodes_clockwise[(highest_node_index + index_offset) % len(connected_nodes_clockwise)]
                                    for index_offset in range(len(connected_nodes_clockwise))]
        # Set a global list containing the binary ground truths provided
        self.ground_truths = ground_truths

    def get_node(self, identifier: int) -> SurfaceTopologyNode:
        """Return a reference to a node given its identifier"""
        # Search for the node with this ID rather than just using the index; they may be out of order
        # There should only be one, so taking the first element is fine
        return [node for node in self.nodes if node.identifier == identifier][0]
