#!/usr/bin/env python3
"""A program for generating realistic audio data based on existing recordings, by tweaking the overall volume as well as the frequency distribution"""
# Created by Brendon Matusch, June 2018

import os
import sys

from numpy.fft import fft

from load_audio import load_audio
from verify_arguments import verify_arguments

# A folder containing the real audio samples in WAV format is expected
verify_arguments('WAV audio sample folder')

# Load each of the audio files from the provided folder and run Fourier transforms on them
audio_recordings_frequency = [fft(audio_recording)
                              for audio_recording in load_audio(sys.argv[1])]
