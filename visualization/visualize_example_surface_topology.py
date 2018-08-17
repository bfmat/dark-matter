#!/usr/bin/env python3
"""A script for loading an example topology and plotting the nodes and the connections between them"""
# Created by Brendon Matusch, August 2018

from data_processing.surface_topology import SurfaceTopologySet

# Load the example JSON data set
topology = SurfaceTopologySet('example_surface_topology_set.json')
