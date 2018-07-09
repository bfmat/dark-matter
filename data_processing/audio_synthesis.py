"""Code for generating modified copies of audio recordings, in order to make a neural network insensitive to various properties of the audio including noise and overall volume"""
# Created by Brendon Matusch, July 2018

import numpy as np


def normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize an audio recording so its geometric mean (standard deviation off of 0) is equal to 1"""
    # Flatten the array and compute the geometric mean (the square root of the average squared value)
    audio_flat_squared = audio.flatten() ** 2
    geometric_mean = np.sqrt(np.mean(audio_flat_squared))
    # Divide the array by the geometric mean to normalize it to that range
    return audio / geometric_mean
