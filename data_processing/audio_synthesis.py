"""Code for generating modified copies of audio recordings, in order to make a neural network insensitive to various properties of the audio including noise and overall volume"""
# Created by Brendon Matusch, July 2018

import numpy as np

from data_processing.audio_domain_processing import time_to_frequency_domain, frequency_to_time_domain


def normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize an audio recording so its geometric mean (standard deviation off of 0) is equal to 1"""
    # Flatten the array and compute the geometric mean (the square root of the average squared value)
    audio_flat_squared = audio.flatten() ** 2
    geometric_mean = np.sqrt(np.mean(audio_flat_squared))
    # Divide the array by the geometric mean to normalize it to that range
    return audio / geometric_mean


def add_time_noise(audio: np.ndarray, standard_deviation: float) -> np.ndarray:
    """Add Gaussian noise to the time domain representation of a recording, so that each sample is adjusted slightly"""
    # Create Gaussian noise centered at 0 and with the provided standard deviation, with the size of the audio array
    noise = np.random.normal(
        loc=0,
        scale=standard_deviation,
        size=audio.shape
    )
    # Return a copy of the audio array with the noise added
    return audio + noise


def multiply_frequency_noise(audio: np.ndarray, standard_deviation: float) -> np.ndarray:
    """Incorporate multiplicative Gaussian noise into an audio recording, such that frequency domain values are modified randomly proportionately to their absolute magnitudes"""
    # Convert the audio into the frequency domain
    audio_frequency = time_to_frequency_domain(audio)
    # Create a noise array of the same size, centered at 1 (because it is multiplicative)
    noise = np.random.normal(
        loc=1,
        scale=standard_deviation,
        size=audio_frequency.shape
    )
    # Element-wise multiply the audio array by the noise
    audio_frequency *= noise
    # Return the modified audio, converted back to the time domain
    return frequency_to_time_domain(audio_frequency)
