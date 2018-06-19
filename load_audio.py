"""A function for loading a folder of audio files in WAV format"""
# Created by Brendon Matusch, June 2018

import os
from typing import List

import numpy as np
from scipy.io.wavfile import read


def load_audio(folder_path: int) -> List[np.ndarray]:
    """Load a number of audio files in WAV format"""
    # Get the full paths of the audio files in the folder
    audio_folder = os.path.expanduser(folder_path)
    audio_paths = [
        os.path.join(audio_folder, file_name)
        for file_name in os.listdir(audio_folder)
    ]
    # Load each of the files, keeping the data but ignoring the sample rate
    # Also, take the first element of each sub-array, so that the output is one-dimensional (mono rather than stereo)
    return [read(path)[1][:, 0] for path in audio_paths]
