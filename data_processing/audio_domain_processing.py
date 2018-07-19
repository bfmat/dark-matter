"""Code related to the division of audio into bands and conversion of audio data from the domain to the frequency domain and vice versa"""
# Created by Brendon Matusch, June 2018

from typing import List

import numpy as np


def time_to_frequency_domain(time_domain_audio: np.ndarray) -> np.ndarray:
    """Convert a time domain audio recording, with an arbitrary number of channels, into a frequency domain recording with the same length"""
    # Transpose the audio array so that the channels axis comes first
    channels_first = np.transpose(time_domain_audio)
    # Run a real Fourier transform on each of the channels, which omits the negative part of the spectrum
    # This is because there is no new information; for real inputs, the output is Hermitian, meaning the negative terms are the complex conjugates of the corresponding real terms
    channels_first_frequency = np.stack([
        np.fft.rfft(single_channel)
        for single_channel in channels_first
    ])
    # Transpose it back so that the sample axis comes first, and return it
    return np.transpose(channels_first_frequency)


def frequency_to_time_domain(frequency_domain_audio: np.ndarray) -> np.ndarray:
    """Convert a frequency domain audio recording, with an arbitrary number of channels, into a time recording with the same length"""
    # Transpose the audio array so that the channels axis comes first
    channels_first = np.transpose(frequency_domain_audio)
    # Run an inverse real Fourier transform on each of the channels
    channels_first_time = np.stack([
        np.fft.irfft(single_channel)
        for single_channel in channels_first
    ])
    # Transpose it back so that the sample axis comes first, and return it
    return np.transpose(channels_first_time)


def band_time_domain(time_domain_audio: np.ndarray, bands: int) -> List[np.ndarray]:
    """Split a time domain audio recording into a specified number of bands in a list"""
    # Get the indices that separate each of the bands, including the beginning and end
    band_separators = np.linspace(
        start=0,
        stop=time_domain_audio.shape[0],
        num=bands + 1
    )
    # Round the band separators to integers
    band_separators_integer = [
        int(round(separator))
        for separator in band_separators
    ]
    # Split the recording along the samples axis according to the separator indices, and return the resulting list
    return [
        time_domain_audio[
            band_separators_integer[band_index]:band_separators_integer[band_index + 1]
        ]
        for band_index in range(bands)
    ]
