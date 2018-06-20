"""A function for loading a folder of audio files in WAV format"""
# Created by Brendon Matusch, June 2018

import os
from typing import List, Tuple

import numpy as np
from scipy.io.wavfile import read


def load_audio(folder_path: str) -> List[Tuple[int, np.ndarray]]:
    """Load a number of audio files in WAV format"""
    # Get the full paths of the audio files in the folder
    audio_folder = os.path.expanduser(folder_path)
    audio_paths = [
        os.path.join(audio_folder, file_name)
        for file_name in os.listdir(audio_folder)
    ]
    # Load each of the files, including the sample rates as well as the data in NumPy array format
    rates, data_arrays = zip(*(read(path) for path in audio_paths))
    # Take the first element of each multi-dimensional sub-array in the data array so that the output is one-dimensional (mono rather than stereo)
    data_arrays_mono = [
        array[:, 0] if len(array.shape) > 1 else array
        for array in data_arrays
    ]
    # Zip the rates with the data arrays and return them
    return list(zip(rates, data_arrays_mono))
