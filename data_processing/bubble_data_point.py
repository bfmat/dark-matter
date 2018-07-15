"""Code related to the custom bubble data format used instead of ROOT"""
# Created by Brendon Matusch, June 2018

from enum import Enum, IntEnum, auto
import datetime
import itertools
import math
import os
import random
import sys
from typing import List, Optional

from data_processing.audio_domain_processing import time_to_frequency_domain, band_time_domain, band_frequency_domain
from data_processing.audio_synthesis import normalize, add_time_noise, multiply_frequency_noise

import numpy as np
from scipy.ndimage import imread

# The path in which all of the raw images and audio data are stored
RAW_DATA_PATH = os.path.expanduser('~/projects/rrg-kenclark/pico/30l-16-data')

# The side length in pixels of the square windows to crop out of the bubble chamber images
WINDOW_SIDE_LENGTH = 50
# The number of images to load out of the 21 possibilities for each bubble
IMAGES_PER_BUBBLE = 1

# The number of piezo channels present in the audio files (not all of them work)
CHANNELS = 8
# The number of times to multiply distinct noise into the frequency domain when synthesizing audio training examples
FREQUENCY_DOMAIN_NOISE_COUNT = 4
# The number of times to add distinct noise to the time domain for each frequency domain example
TIME_DOMAIN_NOISE_COUNT = 4
# The standard deviation off of 1, after normalization, for multiplicative noise to the frequency domain
FREQUENCY_DOMAIN_NOISE_STANDARD_DEVIATION = 0.1
# The standard deviation, after normalization, for additive noise to the time domain
TIME_DOMAIN_NOISE_STANDARD_DEVIATION = 0.05

# The number of bands to split the recording into for frequency domain processing, from beginning to end
TIME_BANDS = 16
# The number of bands to split the frequency domain representation of each time band into
FREQUENCY_BANDS = 1024


class RunType(Enum):
    """An enumeration of the possible run types, including various distinct radiation sources; numbers correspond to the numeric representations in the data table"""
    # Run with normal background radiation
    LOW_BACKGROUND = auto()
    # Calibration with AmBe neutron source at the end of the source tube
    AMERICIUM_BERYLLIUM = auto()
    # Californium neutron source at various distances from the bottom of the source tube
    CALIFORNIUM = auto()
    # Barium gamma source at various distances from the bottom of the source tube
    BARIUM = auto()
    # Cobalt gamma source at 45cm from the bottom of the source tube
    COBALT = auto()
    # Engineering, test and commissioning data which should not be used
    GARBAGE = auto()


# A dictionary mapping numeric run types to values in the enum
RUN_TYPE_DICTIONARY = {
    # Unblind background radiation data
    0: RunType.LOW_BACKGROUND,
    # Data with acoustics originally blinded, with a 200Hz camera frame rate
    10: RunType.LOW_BACKGROUND,
    # More data with acoustics originally blinded, with a 340Hz camera frame rate
    100: RunType.LOW_BACKGROUND,
    # Neutron calibration data with AmBe source at the bottom of the source tube
    2: RunType.AMERICIUM_BERYLLIUM,
    # Neutron calibration data with 252Cf source 40cm from the bottom of the source tube
    14: RunType.CALIFORNIUM,
    # Neutron calibration data with 252Cf source 60cm from the bottom of the source tube
    15: RunType.CALIFORNIUM,
    # Neutron calibration data with 252Cf source 80cm from the bottom of the source tube
    16: RunType.CALIFORNIUM,
    # Gamma calibration data with 133Ba source 100cm from the bottom of the source tube
    21: RunType.BARIUM,
    # Gamma calibration data with 133Ba source 45cm from the bottom of the source tube
    22: RunType.BARIUM,
    # Gamma calibration data with 133Ba source 120cm from the bottom of the source tube
    23: RunType.BARIUM,
    # Gamma calibration data with 133Ba source 10cm from the bottom of the source tube
    24: RunType.BARIUM,
    # Gamma calibration data with 133Ba source 45cm from the bottom of the source tube, after the frame rate change
    32: RunType.BARIUM,
    # Gamma calibration data with 60Co source 45cm from the bottom of the source tube, after the frame rate change
    41: RunType.COBALT,
    # Engineering, test and commissioning data which should not be used
    99: RunType.GARBAGE
}


class TriggerCause(IntEnum):
    """An enumeration of the possible causes of the recording trigger"""
    CAMERA_TRIGGER = 0  # The normal optical trigger when bubbles are observed
    TIMEOUT = 2  # The maximum time permitted for an event was reached
    MANUAL_OR_RELAUNCH = 3  # Either a manual trigger, or an auto-relaunch due to a problem


class BubbleDataPoint:
    """A bubble event data class that contains all of the data necessary for training, for a single bubble; there are multiple of these recorded for a multi-bubble event"""

    def __init__(self, root_event, unique_bubble_index: int) -> None:
        """Initializer that takes a single event in the CERN ROOT format and extracts the relevant data; it also takes a unique index for this bubble"""
        # Set a global variable with the provided unique index
        self.unique_bubble_index = unique_bubble_index
        # Get the timestamp that this event was recorded at
        self.timestamp = root_event.timestamp
        # The run identifier is in the format YYYYMMDD_RR (R is the run within that day); parse it to get a date and a run number
        run_identifier = root_event.run
        year = int(run_identifier[:4])
        month = int(run_identifier[4:6])
        day = int(run_identifier[6:8])
        self.date = datetime.date(year=year, month=month, day=day)
       # At the end of the string, there are often strange characters; iterate up to the last one which is a numeric digit, and use that to cut out the run number
        run_number_end_index = 10
        for character_index in itertools.count(run_number_end_index + 1):
            # Subtract one from the index because indexing in Python is exclusive for the end
            if run_identifier[character_index - 1].isdigit():
                run_number_end_index = character_index
            else:
                break
        self.run_number = int(run_identifier[9:run_number_end_index])
        # Get the event number within the run, which starts at 0
        self.event_number = root_event.ev
        # Assign the run type based on the raw numeric value, assuming garbage if the value is not present in the enumeration
        raw_run_type = root_event.run_type
        self.run_type = RunType(raw_run_type) \
            if raw_run_type in set(possible_run_type.value for possible_run_type in RunType) \
            else RunType.GARBAGE
        # Likewise for the cause of the recording trigger, assuming a manual trigger or relaunch due to a problem
        raw_trigger_cause = root_event.trigger_main
        self.trigger_cause = TriggerCause(raw_trigger_cause) \
            if raw_trigger_cause in set(possible_trigger_cause.value for possible_trigger_cause in TriggerCause) \
            else TriggerCause.MANUAL_OR_RELAUNCH
        # Get the position-corrected banded frequency domain representation of the audio as an array of strength values
        # It has to be converted to a list first; NumPy reads the length incorrectly
        banded_array = np.array(list(root_event.piezo_E_PosCor))
        # Reshape it into the format (time bin, frequency bin, piezo channel) where there are 3 time bins, 8 frequency bins, and 3 piezo channels
        self.banded_frequency_domain = np.reshape(banded_array, (3, 8, 3))
        # Get the approximated position of the bubble in 3 dimensions
        self.x_position = root_event.X
        self.y_position = root_event.Y
        self.z_position = root_event.Z
        # Get the distance from the center of the jar from the perspective of a camera, replacing it with a very large value if it is negative (could not be found)
        if root_event.R2 >= 0:
            self.distance_from_center = math.sqrt(root_event.R2)
        else:
            self.distance_from_center = 100000
        # Get the minimum of the distances from the bubble to the wall on two axes
        self.distance_to_wall = min(root_event.Dwall, root_event.Dwall_horiz)
        # Compute the logarithmic acoustic parameter, which is used to sort background events out of the calibration runs
        # Substitute a large negative number if the raw value is zero or negative
        self.logarithmic_acoustic_parameter = math.log(root_event.acoustic_bubnum, 10) if root_event.acoustic_bubnum > 0 \
            else -10_000
        # Get the X and Y positions of the bubble on each of the four cameras
        self.camera_positions = [
            (root_event.hori0, root_event.vert0),
            (root_event.hori1, root_event.vert1),
            (root_event.hori2, root_event.vert2),
            (root_event.hori3, root_event.vert3)
        ]
        # Get the number of bubbles present in the event, calculated through image matching
        self.num_bubbles_image = root_event.nbub
        # Get the approximated number of bubbles based on the pressure transducer
        self.num_bubbles_pressure = root_event.dytranCZ
        # Get the neural network score predicted in the original PICO-60 paper
        self.original_neural_network_score = root_event.NN_score


def bubble_data_path(bubble: BubbleDataPoint) -> str:
    """Given a bubble data point, return the path to the folder containing all of the data corresponding to that bubble"""
    # Format the date and run number to get the folder name, padding one-digit month and day numbers with zeroes
    date_run_folder_name = f'{bubble.date.year}{bubble.date.month:0>2d}{bubble.date.day:0>2d}_{bubble.run_number}'
    # Paths to the image folders are inconsistent; sometimes they contain another folder named with the date inside the first folder, and sometimes they do not
    # Check if there is a folder inside the first one and format the path accordingly
    date_run_folder_path = os.path.join(RAW_DATA_PATH, date_run_folder_name)
    name_duplicated_path = os.path.join(
        date_run_folder_path, date_run_folder_name)
    if os.path.isdir(name_duplicated_path):
        date_run_folder_path = name_duplicated_path
    # Combine this with the event number to get the full path to the data folder
    return os.path.join(
        date_run_folder_path,
        str(bubble.event_number)
    )


def load_bubble_audio(bubble: Optional[BubbleDataPoint], audio_file_path: Optional[str] = None, use_synthesis: bool = False) -> List[np.ndarray]:
    """Load an audio file in the raw binary format present in the PICO-60 data set, returning an array with dimensions the number of samples containing the data from only the working microphones by 2, or multiple arrays with noise added if synthesis is enabled"""
    # If a path is not provided, get it from the bubble
    if audio_file_path is None:
        # Get the path to the bubble data folder, and load the audio binary file within it
        audio_file_path = os.path.join(
            bubble_data_path(bubble),
            'fastDAQ_0.bin'
        )
    # Try to open the file for binary reading
    try:
        with open(audio_file_path, 'rb') as audio_file:
            # Read and ignore 4 bytes which comprise the header
            audio_file.read(4)
            # The next 2 bytes are the length of the string describing the channels
            channels_string_length = int.from_bytes(
                audio_file.read(2), sys.byteorder)
            # Read the channels description string from the file now, and decode it as a string
            channels_string = audio_file.read(channels_string_length).decode()
            # Read the number of samples from the file, which is used in parsing the rest of the file
            samples = int.from_bytes(audio_file.read(4), sys.byteorder)
            # The rest of the file consists of 2-byte integers, one per channel per sample; read all of it
            raw_data = audio_file.read(CHANNELS * samples * 2)
    # If we are not permitted to load the file or if it does not exist, notify the user and return an empty list with no examples
    except (PermissionError, FileNotFoundError):
        print(f'File {audio_file_path} could not be loaded')
        return []
    # Convert the data into a 1-dimensional NumPy array
    data_array_flat = np.frombuffer(raw_data, dtype=np.int16)
    # Reshape the 1-dimensional array into channels and samples
    data_array = np.reshape(data_array_flat, (samples, CHANNELS))
    # Index and return the data of microphones 3 and 7, the only ones that work
    data_array = data_array[:, [0, 3]]
    # Cut out most of the empty noise at the beginning, and the hydraulic pump sounds at the end
    data_array = data_array[90_000:190_000]
    # Return the data array without any further processing (temporary)
    return [data_array]
    # # Convert the array to 64-bit floats to prevent overflows
    # data_array = data_array.astype(np.float64)
    # # Normalize the audio array so its geometric mean is 1
    # data_array = normalize(data_array)
    # # If synthesis is not enabled, wrap it in a single-element list because that is expected by the data generator and return it
    # if not use_synthesis:
    #     return [data_array]
    # # Otherwise, create a list to add synthesized examples to (starting with the real example) and iterate over the number of times to multiply noise into the frequency domain
    # examples = [data_array]
    # for _ in range(FREQUENCY_DOMAIN_NOISE_COUNT):
    #     # Incorporate frequency domain noise into the audio with the defined standard deviation and add it to the list
    #     audio_with_frequency_noise = multiply_frequency_noise(
    #         audio=data_array,
    #         standard_deviation=FREQUENCY_DOMAIN_NOISE_STANDARD_DEVIATION
    #     )
    #     examples.append(audio_with_frequency_noise)
    #     # Iterate over the number of times to add noise to the time domain
    #     for _ in range(TIME_DOMAIN_NOISE_COUNT):
    #         # Add time domain noise to the audio that already has frequency noise and add it to the list
    #         audio_with_time_noise = add_time_noise(
    #             audio=audio_with_frequency_noise,
    #             standard_deviation=TIME_DOMAIN_NOISE_STANDARD_DEVIATION
    #         )
    #         examples.append(audio_with_time_noise)
    # # Return the list including the original examples alongside synthesized examples
    # return examples


def load_bubble_frequency_domain(bubble: BubbleDataPoint, use_synthesis: bool = False) -> List[np.ndarray]:
    """Given a bubble data point, load a flattened representation of the audio in frequency domain in various time and frequency bins and for both channels"""
    # Note that synthesis is currently not implemented for this loading function, but is a necessary part of the API
    # First, get the audio waveform corresponding to this bubble
    time_domain = load_bubble_audio(bubble)[0]
    # Split into a list of multiple time bands
    time_domain_bands = band_time_domain(time_domain, TIME_BANDS)
    # Convert each of the time bands into frequency domain
    time_bands_frequency = [
        time_to_frequency_domain(band)
        for band in time_domain_bands
    ]
    # Reduce the frequency domain representations into bands and stack them together into a single array
    banded_frequency_domain = np.stack([
        band_frequency_domain(band, FREQUENCY_BANDS)
        for band in time_bands_frequency
    ])
    # Flatten the array so it can be input into a dense layer and return the result wrapped in a single-element list
    return [banded_frequency_domain.flatten()]


def load_bubble_images(bubble: BubbleDataPoint) -> List[np.ndarray]:
    """Given a bubble data point, load, crop, and return a list of windows which contain that bubble"""
    # Get the path to the bubble data folder and use it to get that to the image folder
    image_folder_path = os.path.join(
        bubble_data_path(bubble),
        'Images'
    )
    # Create a list to hold the images of this bubble
    bubble_images = []
    # Iterate over the number of each of the four cameras and the position of the bubble in this camera
    for camera_number, (bubble_x, bubble_y) in enumerate(bubble.camera_positions):
        # If either axis is less than zero, the bubble could not be found; skip to the next iteration
        if bubble_x < 0 or bubble_y < 0:
            continue
        # Otherwise, iterate over randomly chosen samples from the image numbers 50 to 70 inclusive, which are recorded after the bubble is detected
        for image_number in random.sample(range(50, 71), IMAGES_PER_BUBBLE):
            # Format the full path of this image
            image_path = os.path.join(
                image_folder_path,
                f'cam{camera_number}_image{image_number}.png'
            )
            # Attempt to load it as a NumPy array, skipping to the next iteration if it we are not permitted to load the image or if it does not exist
            try:
                full_image = imread(image_path)
            except (PermissionError, FileNotFoundError):
                print(f'File {image_path} could not be loaded')
                continue
            # Round the bubble X and Y positions to integers and cap them at the edges of the image, so they can be used to index the image
            bubble_x_integer, bubble_y_integer = (
                min(max(int(round(position)), WINDOW_SIDE_LENGTH),
                    shape_dimension - (WINDOW_SIDE_LENGTH + 1))
                for position, shape_dimension in zip([bubble_x, bubble_y], reversed(full_image.shape))
            )
            # Crop a square out, centered at the integer position (the image is indexed Y, X)
            half_side_length = WINDOW_SIDE_LENGTH // 2
            window = full_image[(bubble_y_integer - half_side_length):(bubble_y_integer + half_side_length),
                                (bubble_x_integer - half_side_length):(bubble_x_integer + half_side_length)]
            # Add a channel dimension of 1 at the end; it is expected by Keras
            window = np.expand_dims(window, axis=-1)
            # Add the cropped window to the list of images
            bubble_images.append(window)
    # Return the list of cropped images
    return bubble_images
