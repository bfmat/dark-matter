#!/usr/bin/env python3
"""A program for generating realistic audio data based on existing recordings, by tweaking the overall volume as well as the frequency distribution"""
# Created by Brendon Matusch, June 2018

import os
import sys
import itertools

import numpy as np
from scipy.io.wavfile import write

from data_processing.load_audio import load_audio
from utilities.verify_arguments import verify_arguments

# A folder containing the audio files in WAV format is expected, along with various parameters for disturbances to the audio
verify_arguments('WAV audio folder', 'output folder', 'number of outputs per input', 'average volume offset',
                 'volume offset standard deviation', 'individual sample offset standard deviation')

# Assign the command line arguments to variables
wav_audio_folder, output_folder = sys.argv[1:3]
num_outputs_per_input = int(sys.argv[3])
average_volume_offset, volume_offset_std_dev, individual_sample_offset_std_dev = \
    (float(argument) for argument in sys.argv[4:])

# Create a list to add the modified audio recordings to, in the format of tuples with a sample rate followed by a data array
modified_recordings = []
# Load each of the audio files from the provided folder and iterate over the sample rates and data recordings
for sample_rate, audio_recording in load_audio(wav_audio_folder):
    # Run a Fourier transform on the recording to get it in the frequency domain
    audio_recording_frequency = np.fft.fft(audio_recording)
    # Iterate over the number of outputs per input, creating a modified recording on each iteration
    for _ in range(num_outputs_per_input):
        # Get an offset for the volume of the audio from a normal distribution centered at the average offset, and add it to a copy of the recording in frequency domain
        volume_offset = np.random.normal(
            loc=average_volume_offset,
            scale=volume_offset_std_dev
        )
        modified_recording = audio_recording_frequency + volume_offset
        # Offset each of the samples individually with a normal distribution centered at 0
        modified_recording += np.random.normal(
            loc=0,
            scale=individual_sample_offset_std_dev,
            size=len(modified_recording)
        )
        # Add the resulting recording alongside the sample rate to the list
        modified_recordings.append((sample_rate, modified_recording))

# Iterate over the modified recordings to save them, with corresponding numbers for file names
for (sample_rate, recording), file_number in zip(modified_recordings, itertools.count()):
    # Run a reverse Fourier transform to get it back in the time domain
    recording_time = np.fft.ifft(recording)
    # Convert it from complex to real so it can be saved as audio
    recording_time_real = np.real(recording_time)
    # Save it as an audio file with a numbered name in the output folder
    output_file_path = os.path.join(output_folder, f'{file_number}.wav')
    write(output_file_path, sample_rate, recording_time_real)
