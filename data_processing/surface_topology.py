"""Code related to a data structure containing the triangulated surface of a 2-dimensional object, with a set of numeric neural network input values corresponding to each triangle"""
# Created by Brendon Matusch, August 2018

import json
import os


class SurfaceTopologyNode:
    """A single topology node, which contains information loaded from one dictionary in the JSON file"""

    # Placeholders for references to the nodes above, right and left (triangular directions)
    node_above = None
    node_right = None
    node_left = None

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
            # Get the nodes this one is connected to, leaving None as it is
            connected_nodes_clockwise = [self.nodes[node_index] if node_index is not None else None for node_index in node.raw_connections_clockwise]
            # Get the Y (vertical) positions of these connected nodes
            # TODO
            print(connected_nodes_clockwise)



# Temporary: load the example JSON file
topology = SurfaceTopologySet('example_surface_topology_set.json')
