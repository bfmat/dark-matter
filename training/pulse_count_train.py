#!/usr/bin/env python3
"""Train a fully connected neural network on the numbers of pulses for each PMT in the DEAP data"""
# Created by Brendon Matusch, August 2018

from data_processing.load_deap_data import load_deap_data

# Load all events from the file
neck_events, non_neck_events = load_deap_data()
