#!/usr/bin/env python3
"""A script for training a neural network to predict the position of a bubble based on multiple audio waveforms from different piezos"""
# Created by Brendon Matusch, July 2018

from models.waveform_localization_network import create_model

# Create an instance of the waveform localization network
model = create_model()
