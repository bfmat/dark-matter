"""Code related to a data structure containing the triangulated surface of a 2-dimensional object, with a set of numeric neural network input values corresponding to each triangle"""
# Created by Brendon Matusch, August 2018

import json
import os


class SurfaceTopologySet:
    """A data structure containing the triangulated surface of a 2-dimensional object, with a set of numeric neural network input values corresponding to each triangle"""

    def __init__(self, json_path: str):
        """Create a surface topology set, given a path to a JSON file containing the relevant data"""
        # Open the JSON file and load all data out of it
        with open(os.path.expanduser(json_path)) as json_file:
            json_data = json.load(json_file)
        print(json_data)


# Temporary: load the example JSON file
topology = SurfaceTopologySet('example_surface_topology_set.json')
