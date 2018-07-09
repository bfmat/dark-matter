"""Code for generating modified copies of audio recordings, in order to make a neural network insensitive to various properties of the audio including noise and overall volume"""
# Created by Brendon Matusch, July 2018

import numpy as np

from data_processing.audio_domain_processing import time_to_frequency_domain, frequency_to_time_domain, band_time_domain


def normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize an audio recording so its geometric mean (standard deviation off of 0) is equal to 1"""
    # Flatten the array and compute the geometric mean (the square root of the average squared value)
    audio_flat_squared = np.square(audio.flatten())
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


def multiply_frequency_noise(audio: np.ndarray, standard_deviation: float, time_bands: int = 1) -> np.ndarray:
    """Incorporate multiplicative Gaussian noise into an audio recording, such that frequency domain values are modified randomly proportionately to their absolute magnitudes"""
    # Split the audio into a number of time bands (a larger number of bands means more localized application of frequency domain noise)
    audio_bands = band_time_domain(audio, time_bands)
    # Create a list to add modified bands to, and iterate over the original bands
    modified_audio_bands = []
    for audio_band in audio_bands:
        # Convert the audio band into the frequency domain
        audio_frequency = time_to_frequency_domain(audio_band)
        # Create a noise array of the same size, centered at 1 (because it is multiplicative)
        noise = np.random.normal(
            loc=1,
            scale=standard_deviation,
            size=audio_frequency.shape
        )
        # Element-wise multiply the audio array by the noise
        audio_frequency *= noise
        # Convert the modified band back into the time domain
        modified_band = frequency_to_time_domain(audio_frequency)
        # Add the modified band to the list
        modified_audio_bands.append(modified_band)
    # Return the modified bands, concatenated together along the samples axis
    return np.concatenate(modified_audio_bands, axis=0)
