"""A system for projecting 3D PMT pulse count data onto a 2D map using an approximate Mercator map projection"""
# Created by Brendon Matusch, September 2018

import numpy as np

from data_processing.pmt_positions import X_POSITIONS, Y_POSITIONS, Z_POSITIONS


def pmt_map_projection(pulse_counts: np.ndarray) -> np.ndarray:
    """Project a list of pulse counts, at predefined positions along the sphere, onto a 2D map"""
    # Get the unique Z positions (the PMTs are arranged into circular rows)
    row_positions = np.unique(Z_POSITIONS)
    print(row_positions)
    # Iterate over the pulse counts alongside positions in 3D space
    for pulse_count, x_position, y_position, z_position in zip(pulse_counts, X_POSITIONS, Y_POSITIONS, Z_POSITIONS):
        # Get the index of the row that this position belongs to (which corresponds to the Y position in the resulting map)
        row_index = np.where(row_positions == z_position)[0][0]
        print(row_index)


import random
from data_processing.load_deap_data import load_simulated_deap_data
neck_events, _ = load_simulated_deap_data()
neck_event = random.choice(neck_events)
neck_pulses = neck_event[0]
pmt_map_projection(neck_pulses)
