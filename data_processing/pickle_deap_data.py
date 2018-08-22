#!/usr/bin/env python2
"""Load the DEAP Monte Carlo simulation data from ROOT files and save in a portable scikit-learn Joblib format"""
# Created by Brendon Matusch, August 2018
# Written in Python 2 because the DEAP-specific RAT distribution required for loading the relevant files binds to Python 2

from __future__ import print_function

import os

import numpy as np
# Import RAT even though it is not used directly, because it modifies ROOT
import rat
import ROOT

# The total number of PMTs, which is used as the length of various arrays
PMT_COUNT = 255

# Load the data file and get the main tree
data_file = ROOT.TFile(os.path.expanduser('~/PB_000000_analyzed_0100.root'))
tree = data_file.Get('T')
# Create a list to add all PMT photon count and timing arrays to
pmt_data_arrays = []
# Iterate over all events within the tree
for event in tree:
    # Create arrays to hold the integer counts of photons and pulses for each PMT, defaulting to 0 for PMTs that are not included
    photon_counts = np.zeros(PMT_COUNT, dtype=int)
    pulse_counts = np.zeros(PMT_COUNT, dtype=int)
    # Create a list to hold lists of pulse start and end times
    pulse_timings = []
    # Initialize it with an empty list for each PMT
    for _ in range(PMT_COUNT):
        pulse_timings.append([])
    # Iterate over the simulated data objects corresponding to each of the PMTs that were affected by this event
    for pmt_data in event.ds.mc.pmt:
        # Get the identifier of this PMT, which is used as an index because they are not in order
        pmt_identifier = pmt_data.GetID()
        # Set the counts of photons and pulses in the arrays
        photon_counts[pmt_identifier] = pmt_data.GetMCPhotonCount()
        pulse_counts[pmt_identifier] = pmt_data.GetMCPulseCount()
        # Iterate over each of the pulses received by this PMT
        for pulse in pmt_data.pulse:
            # Add the start time of this pulse to the list corresponding to this PMT
            pulse_timings[pmt_identifier].append(pulse.GetStartTime())
    # Add the photon and pulse counts, together with the pulse timings, to the list for all events
    pmt_data_arrays.append((photon_counts, pulse_counts, pulse_timings))
