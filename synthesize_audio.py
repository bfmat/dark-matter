#!/usr/bin/env python3
"""A program for generating realistic audio data based on existing recordings, by tweaking the overall volume as well as the frequency distribution"""
# Created by Brendon Matusch, June 2018

import numpy as np
from scipy.io.wavfile import read
from verify_arguments import verify_arguments

# A folder containing the real audio samples is expected
verify_arguments('audio sample folder')
