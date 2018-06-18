#!/usr/bin/env python3
"""A program for generating realistic audio data based on existing recordings, by tweaking the overall volume as well as the frequency distribution"""
# Created by Brendon Matusch, June 2018

import os
import sys
import numpy as np
from scipy.io.wavfile import read
from verify_arguments import verify_arguments

# A folder containing the real audio samples in WAV format is expected
verify_arguments('audio sample folder')

# Get the full paths of the files in the audio sample folder
sample_folder = os.path.expanduser(sys.argv[1])
audio_paths = [
    os.path.join(sample_folder, file_name)
    for file_name in os.listdir(sample_folder)
]

# TODO: Read the audio and modify it
