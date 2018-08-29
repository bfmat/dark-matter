#!/usr/bin/env python2
"""Load the DEAP simulation and real-word data from ROOT files and save in a portable scikit-learn Joblib format"""
# Created by Brendon Matusch, August 2018
# Written in Python 2 because the DEAP-specific RAT distribution required for loading the relevant files binds to Python 2

from __future__ import print_function

import glob
import itertools
import os

import numpy as np
# Import RAT even though it is not used directly, because it modifies ROOT
import rat
import ROOT
from sklearn.externals import joblib

# The paths the simulated and real-world DEAP data is saved in
SIMULATED_PATH = os.path.expanduser('~/deap_data.pkl')
REAL_WORLD_PATH = os.path.expanduser('~/real_deap_data.pkl')

# The total number of PMTs, which is used as the length of various arrays
PMT_COUNT = 255


def load_data_from_file(file_path, cut_position):
    """Given a path to a ROOT file, load all of the relevant data from it and return it as a list of tuples"""
    # Load the data file from the full path, and get the main tree
    data_file = ROOT.TFile(file_path)
    tree = data_file.Get('T')
    # Create a list to add all PMT photon count and timing arrays to
    pmt_data_arrays = []
    # Iterate over all events within the tree
    for event in tree:
        # If there are supposed to be cuts done on the position to make sure the event is either near the center or in the neck
        if cut_position:
            # If the event's approximated position is outside the vessel
            if event.ds.ev[0].mblikelihood.pos.Mag() >= 630:
                # Skip this event; it is probably not a neck alpha (and in the case of neutron data, it should be removed anyway to avoid introducing a bias)
                continue
        # Try to get the calibration sub-tree, which is the only data available for real-world events
        # It is a vector with either 0 or 1 elements
        try:
            calibration = event.ds.cal[0]
        # If it is not present, this event was not detected, so it can be skipped
        except IndexError:
            continue
        # Create arrays to hold the integer count of pulses for each PMT, defaulting to 0 for PMTs that are not included
        pulse_counts = np.zeros(PMT_COUNT, dtype=int)
        # Create a list to hold the times of photons observed at each PMT
        photon_timings = []
        # Initialize it with an empty list for each PMT
        for _ in range(PMT_COUNT):
            photon_timings.append([])
        # Iterate over the data objects corresponding to each of the PMTs that were affected by this event
        for pmt_data in calibration.pmt:
            # Get the identifier of this PMT, which is used as an index because they are not in order
            pmt_identifier = pmt_data.GetID()
            # Set the count of pulses in the arrays
            pulse_counts[pmt_identifier] = pmt_data.GetPulseCount()
            # Replace the empty list corresponding to this PMT with the list of photon timings
            photon_timings[pmt_identifier] = list(pmt_data.PEtime)
        # Add the pulse counts and photon timings to the list for all events
        pmt_data_arrays.append((pulse_counts, photon_timings))
    # Notify the user that this file has been loaded
    print('Data loaded from file', file_path)
    # Return the list of tuples containing all of the data
    return pmt_data_arrays


# First, we must load and store the training data from the Monte Carlo simulation
# Create 2 lists: 1 for neck events, and the other for non-neck events
# Load all data out of the relevant paths, chaining all of the lists together
neck_events = list(itertools.chain.from_iterable(
    # For the simulation, do not run any position cuts
    load_data_from_file(path, cut_position=False)
    # There are 2 folders that contain all neck events; chain them together
    for path in itertools.chain(
        glob.iglob(os.path.expanduser('~/MC_forBrendon/AlphaPo210_InnerFlowGuide/*')),
        glob.iglob(os.path.expanduser('~/MC_forBrendon/AlphaPo210_OuterFlowGuide/*'))
    )
))
non_neck_events = list(itertools.chain.from_iterable(
    load_data_from_file(path, cut_position=False)
    for path in glob.iglob(os.path.expanduser('~/MC_forBrendon/NuclearRecoil/*'))
))
# Combine the neck and non-neck events in a tuple, and save them in a Joblib binary file
with open(SIMULATED_PATH, 'wb') as joblib_file:
    joblib.dump((neck_events, non_neck_events), joblib_file)
# Notify the user that it has been saved
print('Simulated data file saved at', SIMULATED_PATH)

# Next, we must load and store a set of relatively pure real-world neck alpha events
real_world_neck_events = list(itertools.chain.from_iterable(
    # Cut on the position to remove Cerenkov events
    load_data_from_file(path, cut_position=True)
    # The data is stored across many files in a folder
    for path in glob.iglob(os.path.expanduser('~/skimmedAlphaforBrendon/*'))
))
# Finally, there is a sample of relatively pure real-world neutron events
real_world_neutron_events = list(itertools.chain.from_iterable(
    # Also cut the position here, to avoid introducing biases
    # Do not cut the position here; events not in the center of the vessel are fine
    load_data_from_file(path, cut_position=False)
    # The data is once again stored across several files in a folder
    for path in glob.iglob(os.path.expanduser('~/skimmedNeutronforBrendon/*'))
))
# Combine the neutron and neck alpha events in a tuple, and save them in another Joblib binary file
with open(REAL_WORLD_PATH, 'wb') as joblib_file:
    joblib.dump((real_world_neck_events, real_world_neutron_events), joblib_file)
# Notify the user that it has been saved
print('Real-world data file saved at', REAL_WORLD_PATH)
