"""Code for creating a surface topology with the DEAP detector data"""
# Created by Brendon Matusch, October 2018

import os

import numpy as np

from data_processing.load_deap_data import load_simulated_deap_data
from data_processing.pmt_positions import X_POSITIONS, Y_POSITIONS, Z_POSITIONS
from data_processing.surface_topology import SurfaceTopologySet


def create_deap_topology() -> SurfaceTopologySet:
    """Create and return a surface topology containing the DEAP detector data"""
    # Load all simulated events from the file
    neck_events, non_neck_events = load_simulated_deap_data()
    # Combine the list of events into one, and take the pulse counts from each
    pulse_counts_by_event = [event[0] for event in neck_events + non_neck_events]
    # Transpose the nested list so they are arranged by node
    pulse_counts_by_node = list(zip(*pulse_counts_by_event))
    # Create a list of ground truths, using the numbers of events (they will not be shuffled here)
    ground_truths = ([1] * len(neck_events)) + ([0] * len(non_neck_events))
    # Zip together the positions of each of the nodes
    positions = list(zip(X_POSITIONS, Y_POSITIONS, Z_POSITIONS))
    # Join together the path to the CSV file containing all of the connections
    csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'deap_connections.csv')
    # Create and return a surface topology using this information
    return SurfaceTopologySet(csv_path, pulse_counts_by_node, positions, ground_truths)
