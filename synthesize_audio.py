#!/usr/bin/env python3
"""A program for generating realistic audio data based on existing recordings, by tweaking the overall volume as well as the frequency distribution"""
# Created by Brendon Matusch, June 2018

import os
import sys
import numpy as np
from verify_arguments import verify_arguments

# A folder containing the real audio samples in WAV format is expected
verify_arguments('WAV audio sample folder')
