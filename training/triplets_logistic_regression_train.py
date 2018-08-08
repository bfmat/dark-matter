#!/usr/bin/env python3
"""A script for training a logistic regression model to separate loud from quiet triplet events"""
# Created by Brendon Matusch, August 2018

import numpy as np
from sklearn.linear_model import LogisticRegression

from data_processing.load_triplet_classification_data import load_triplet_classification_data

# Load the triplet events
loud_events, quiet_events = load_triplet_classification_data()
# Stack an input array of flattened position-corrected 8-band Fourier transforms for all of the events together
inputs = np.stack([event.banded_frequency_domain.flatten() for event in loud_events + quiet_events])
# Make a binary output list, where 1 is loud and 0 is quiet
outputs = np.concatenate([np.ones(len(loud_events)), np.zeros(len(quiet_events))])
# Create a logistic regression classifier
classifier = LogisticRegression()
# Train it on the input and output arrays
classifier.fit(X=inputs, y=outputs)
