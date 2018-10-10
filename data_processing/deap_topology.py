"""Code for creating a surface topology with the DEAP detector data"""
# Created by Brendon Matusch, October 2018

import numpy as np

from data_processing.load_deap_data import load_simulated_deap_data
from data_processing.pmt_positions import X_POSITIONS, Y_POSITIONS, Z_POSITIONS
from data_processing.surface_topology import SurfaceTopologySet


def create_deap_topology() -> SurfaceTopologySet:
    """Create and return a surface topology containing the DEAP detector data"""
    # Load all simulated events from the file
    neck_events, non_neck_events = load_simulated_deap_data()
    # Create a list of ground truths, using the numbers of events (they will not be shuffled here)
    ground_truths = ([1] * len(neck_events)) + ([0] * len(non_neck_events))


# Temporary: run the function for debugging purposes
create_deap_topology()
