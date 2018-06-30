"""Code related to the conversion of audio data from the domain to the frequency domain and vice versa"""
# Created by Brendon Matusch, June 2018

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
